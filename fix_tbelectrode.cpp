/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier, Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>
#include "fix_tbelectrode.h"
#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_hybrid.h"
#include "molecule.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "compute.h"
#include "group.h"
#include "domain.h"
#include "region.h"
#include "random_park.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "thermo.h"
#include "output.h"
#include "neighbor.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <complex>

using namespace std;
using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};

/* ---------------------------------------------------------------------- */

Fixtbel::Fixtbel(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
{
  //printf("Beginin of Fixtbel\n");
  if (narg < 8)
    error->all(FLERR,"Incorrect number of fix tbel arguments {}", narg);

  if (atom->molecular == 2)
    error->all(FLERR,"Fix tbel does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 1;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // required args


  potential = utils::numeric(FLERR,arg[3],false,lmp);
  qsteps = utils::inumeric(FLERR,arg[4],false,lmp);
  qtstep = utils::numeric(FLERR,arg[5],false,lmp);
  hubbardp = utils::numeric(FLERR,arg[6],false,lmp);
  drate = utils::numeric(FLERR,arg[7],false,lmp);
  seed = 25091993;

  if (seed <= 0)
    error->all(FLERR,"Illegal fix kmc seed {}", seed);
  if (reservoir_temperature < 0.0)
    error->all(FLERR,"Illegal fix kmc reservoir temperature {}", reservoir_temperature);

  // read options from end of input line
  regionflag = 0;

  options(narg-8,&arg[8]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // setup of coords and imageflags array

  region_xlo = region_xhi = region_ylo = region_yhi =
    region_zlo = region_zhi = 0.0;
  if (regionflag) {
    if (iregion->bboxflag == 0)
      error->all(FLERR,"Fix kmc region does not support a bounding box");
    if (iregion->dynamic_check())
      error->all(FLERR,"Fix kmc region cannot be dynamic");

    region_xlo = iregion->extent_xlo;
    region_xhi = iregion->extent_xhi;
    region_ylo = iregion->extent_ylo;
    region_yhi = iregion->extent_yhi;
    region_zlo = iregion->extent_zlo;
    region_zhi = iregion->extent_zhi;

    if (region_xlo < domain->boxlo[0] || region_xhi > domain->boxhi[0] ||
        region_ylo < domain->boxlo[1] || region_yhi > domain->boxhi[1] ||
        region_zlo < domain->boxlo[2] || region_zhi > domain->boxhi[2])
      error->all(FLERR,"Fix kmc region extends outside simulation box");

  }

  if (mode == ATOM) natoms_per_molecule = 1;
  else error->all(FLERR,"Fix kmc region does not support molecules");

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;


  nfreaction_attempts = 0.0;

  gcmc_nmax = 0;
  local_gas_list = NULL;
  local_react_list = NULL;
  local_locreact_list = NULL;
  local_reg_list = NULL;
  biasedatoms = NULL;
  electrode = NULL;
  surf = NULL;
  reacte = NULL;
  reactg = NULL;
  prod = NULL;

}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void Fixtbel::options(int narg, char **arg)
{
  if (narg < 0)
    utils::missing_cmd_args(FLERR, "fix kmc", error);

  // defaults

  mode = ATOM;
  regionflag = 0;
  iregion = nullptr;

  int iarg = 0;
  while (iarg < narg) {
  if (strcmp(arg[iarg],"mol") == 0) {
      error->all(FLERR,"Fix kmc does not work with molecules (yet)!");
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, "fix kmc", error);
      iregion = domain->get_region_by_id(arg[iarg+1]);
      if (iregion == nullptr)
        error->all(FLERR,"Region ID for fix kmc does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix kmc command {}", arg[iarg]);
  }
}

/* ---------------------------------------------------------------------- */

Fixtbel::~Fixtbel()
{
  //  printf("Fixtbel()");
  if (regionflag) delete [] idregion;
  delete random_equal;
  delete random_unequal;

  memory->destroy(local_gas_list);
  memory->destroy(local_react_list);
  memory->destroy(local_locreact_list);
  memory->destroy(local_reg_list);
  memory->destroy(atom_coord);
  memory->destroy(coords);
  memory->destroy(imageflags);
  memory->destroy(biasedatoms);
  memory->destroy(electrode);
  memory->destroy(surf);
  memory->destroy(reacte);
  memory->destroy(reactg);
  memory->destroy(prod);
   // if (comm->me == 0) printf("End of Fixtbel::~Fixtbel()\n");
}

/* ---------------------------------------------------------------------- */

int Fixtbel::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void Fixtbel::init()
{
  triclinic = domain->triclinic;

  int *type = atom->type;
  if (mode == ATOM) {
    if (product_type <= 0 || product_type > atom->ntypes)
      error->all(FLERR,"Invalid product atom type in fix kmc command {}", product_type);
    if (reactive_type <= 0 || reactive_type > atom->ntypes)
      error->all(FLERR,"Invalid reactive atom type in fix kmc command {}", reactive_type);
  }

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix kmc in a 2d simulation");

  // create a new group for interaction exclusions

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do kmc on atoms in atom_modify first group");
  }

  imagezero = ((imageint) IMGMAX << IMG2BITS) |
             ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // construct group bitmask for all new atoms
  groupbitall = 1 | groupbit;

  neighbor->add_request(this,NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void Fixtbel::pre_exchange()
{

  if (next_reneighbor != update->ntimestep) return;
  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  if (triclinic) {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  } else {
    sublo = domain->sublo;
    subhi = domain->subhi;
  }

  volume = domain->xprd * domain->yprd * domain->zprd;

  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();

  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  update_gas_atoms_list();
  update_locreact_atoms_list();
  update_reactive_atoms_list();
  update_region_atoms_list();

  //printf("nprod=%d nreact=%d nreg=%i\n", nprod_local, nreact_local, nreg);
        tbsize = atom->nlocal+nreact;
        MatrixXd fock_ab = MatrixXd::Zero(tbsize,tbsize);

        biasedatoms = (int *) memory->smalloc((tbsize)*sizeof(int),
        "GCMC:biasedatoms");
        electrode = (int *) memory->smalloc((tbsize)*sizeof(int),
        "GCMC:electrode");
        surf = (int *) memory->smalloc((tbsize)*sizeof(int),
        "GCMC:surf");
        reactg = (int *) memory->smalloc((tbsize)*sizeof(int),
        "GCMC:reactg");
        reacte = (int *) memory->smalloc((tbsize)*sizeof(int),
        "GCMC:reacte");
        prod = (int *) memory->smalloc((tbsize)*sizeof(int),
        "GCMC:prod");
        for (int ii=0; ii<tbsize; ii++)
        {
           biasedatoms[ii] = 0;
           electrode[ii] = 0;
           surf[ii] = 0;
           reactg[ii] = 0;
           reacte[ii] = 0;
           prod[ii] = 0;
        }
        fill_lists(biasedatoms, electrode, surf, reactg, reacte, prod);
        fill_fock(fock_ab);

        MatrixXd fock_biased = fock_ab;
        applybias(fock_biased, biasedatoms, potential, tbsize);
        MatrixXcd dens_ab = MatrixXcd::Zero(tbsize,tbsize);
        MatrixXcd dens_ref = MatrixXcd::Zero(tbsize,tbsize);

        if (exists_test("density.dat")){
			dens_ab = readmatrix(tbsize);
		}
        else {

			dens_ab = find_gs(fock_ab);
        }

        dens_ref = getgs(fock_biased);
        MatrixXd refshape = stripdensref();
        MatrixXd hubbard = makehubbard(dens_ab);

        //STRIP SYSTEM

        //definir nuevas matrices
        MatrixXd local_fock_ab = localizeMatrixd(fock_ab);
        MatrixXcd local_dens_ab = localizeMatrix(dens_ab);
        MatrixXd local_refshape = localizeMatrixd(refshape);
        MatrixXcd local_dens_ref = localizeMatrix(dens_ref);
        MatrixXd local_hubbard = localizeMatrixd(hubbard);
        //MatrixXd local_fock_ab = fock_ab;
        //MatrixXcd local_dens_ab = dens_ab;
        //MatrixXd local_refshape = refshape;
        //MatrixXcd local_dens_ref = dens_ref;



        for(int ii=0; ii<qsteps; ii++)
        {
		hubbard = makehubbard(dens_ab);
		local_hubbard = hubbardp * localizeMatrixd(hubbard);
        local_dens_ab = rungecuta(local_dens_ab,
          local_fock_ab,
          local_refshape,
          local_hubbard,
          qtstep, //tstep
          drate, //driving rate
          local_dens_ref,
          ii);
        if(ii%413 == 0)
        {
          outstepcharge(local_fock_ab, local_dens_ab, ii);
          printf("%f\n",local_dens_ab.trace().real());
          printf("%f\n",local_dens_ab.trace().imag());
        }
	    }

	    update_density(dens_ab, local_dens_ab);
	    outcharge(dens_ab);

	    write_matrix(dens_ab);

        memory->sfree(biasedatoms);
        memory->sfree(electrode);
        memory->sfree(surf);
        memory->sfree(reactg);
        memory->sfree(reacte);
        memory->sfree(prod);

  next_reneighbor = update->ntimestep+1;
 //if (comm->me == 0) printf("End of Fixtbel::pre_exchange()\n");
}

void Fixtbel::fill_lists(int *biasedatoms,
                          int *electrode,
                          int *surf,
                          int *reactg,
                          int *reacte,
                          int *prod){
  int i, j;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  for(j=0; j<nlocal; j++){
	  //i = local_reg_list[j];
	  if(type[j] == 3){*(biasedatoms+j) = 1;}
	  else if(type[j] == 4){*(electrode+j) = 1;}
      else if(type[j] == 5){*(surf+j) = 1;}
      else if(type[j] == 1){*(reactg+j) = 1;}
      else if(type[j] == 2){*(prod+j) = 1;}
  }
  for(i=nlocal; i<nlocal+nreact; i++){
	  *(reacte+i) = 1;
  }
}
MatrixXd Fixtbel::localizeMatrixd(Ref<MatrixXd> m){
	MatrixXd n = MatrixXd::Zero(nreg+nlocreact,nreg+nlocreact);
	int iglob, jglob;
	int nlocal = atom->nlocal;

	for(int j=0; j<nreg; j++){
		for(int i=0; i<nreg; i++){
			iglob = local_reg_list[i];
			jglob = local_reg_list[j];
			n(j,i) = m(jglob,iglob);
		}
	}
	for(int j=0; j<nlocreact; j++){
		jglob = local_reg_list[nlocal+j];
		n(nreg+j,nreg+j) = m(jglob,jglob);
		for(int i=0; i<nreg; i++){
			iglob = local_reg_list[i];
			n(nreg+j,i) = m(jglob,iglob);
			n(i,nreg+j) = m(iglob,jglob);
		}
	}
	return n;
}

MatrixXcd Fixtbel::localizeMatrix(Ref<MatrixXcd> m){

	MatrixXcd n = MatrixXcd::Zero(nreg+nlocreact,nreg+nlocreact);
	int iglob, jglob;
	int nlocal = atom->nlocal;

	for(int j=0; j<nreg; j++){
		for(int i=0; i<nreg; i++){
			iglob = local_reg_list[i];
			jglob = local_reg_list[j];
			n(j,i) = m(jglob,iglob);
		}
	}
	for(int j=0; j<nlocreact; j++){
		jglob = local_reg_list[nlocal+j];
		n(nreg+j,nreg+j) = m(jglob,jglob);
		for(int i=0; i<nreg; i++){
			iglob = local_reg_list[i];
			n(nreg+j,i) = m(jglob,iglob);
			n(i,nreg+j) = m(iglob,jglob);
		}
	}

	return n;
}

void Fixtbel::update_density(Ref<MatrixXcd> dens_ab, Ref<MatrixXcd> local_dens_ab){

	int iglob, jglob;
	int nlocal = atom->nlocal;

	for(int j=0; j<nreg; j++){
		for(int i=0; i<nreg; i++){
			iglob = local_reg_list[i];
			jglob = local_reg_list[j];
			dens_ab(jglob,iglob) = local_dens_ab(j,i);
		}
	}
	for(int j=0; j<nlocreact; j++){
	    jglob = local_reg_list[nlocal+j];
	    dens_ab(jglob,jglob) = local_dens_ab(nreg+j,nreg+j);
		for(int i=0; i<nreg; i++){
			iglob = local_reg_list[i];
			dens_ab(jglob,iglob) = local_dens_ab(nreg+j,i);
			dens_ab(iglob,jglob) = local_dens_ab(i,nreg+j);
		}
	}
}


void Fixtbel::fill_fock(Ref<MatrixXd> fock_ab){
  //printf("inside fill fock\n");
  double **x = atom->x;
  double dx, dy, dz, dr;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double sitevect[7] = {-1.0, -0.020, -0.10, 0.0, 0.0, 0.0, 0.030};
  int table1[nlocal+nreact];

  int i, j, k;
  k = 0;
  for(i=0; i<nlocal+nreact; i++){
	  if(reactg[i]){
		  table1[i] = nlocal+k;
		  k++;
	  }
  }
  k=0;
  //printf("before for\n");
  for(i = 0; i<nlocal+nreact; i++){
    if(reactg[i]){
		//printf("k=%i nreact=%i nreg=%i\n", k, nreact, nreg);
		fock_ab(i,i) = -0.020;
		fock_ab(i,nlocal+k) = -0.01;
		fock_ab(nlocal+k,i) = -0.01;
		k++;
	}
    else if(reacte[i]){fock_ab(i,i) = 0.030;}
    else if(prod[i]){fock_ab(i,i) = -1.00;}
    //printf("before for 2\n");
    for(j = i+1; j<nlocal+nreact; j++){
		//printf("i=%i  j=%i\n", i,j);
		dx = (x[i][0]-x[j][0]);
		dx = dx - (xhi-xlo) * (int) (dx/(xhi-xlo));
		dy = (x[i][1]-x[j][1]);
		dy = dy - (yhi-ylo) * (int) (dy/(yhi-ylo));
		dz = (x[i][2]-x[j][2]);
		dz = dz - (zhi-zlo) * (int) (dz/(zhi-zlo));
		dr = dx*dx + dy*dy + dz*dz;
		if(dr<16){
			//printf("inside if2\n");
			if(surf[i] && reactg[j]){
				//printf("inside if3\n");
			  dr = 0.01*exp(-1.0*sqrt(dr));
              fock_ab(i,j) = dr;
              fock_ab(j,i) = dr;
              fock_ab(i,table1[j]) = dr;
              fock_ab(table1[j],i) = dr;
	        }
	        else if(surf[j] && reactg[i]){
			  dr = 0.01*exp(-1.0*sqrt(dr));
              fock_ab(i,j) = dr;
              fock_ab(j,i) = dr;
              fock_ab(j,table1[i]) = dr;
              fock_ab(table1[i],j) = dr;
	        }
            else if((biasedatoms[j] || electrode[j] || surf[j]) &&
                (biasedatoms[i] || electrode[i] || surf[i])){
					if(dr<9){
              fock_ab(i,j) = -0.05;
	          fock_ab(j,i) = -0.05;
		    }
            }
        }
	}
  }
}

bool Fixtbel::exists_test (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

MatrixXcd Fixtbel::readmatrix(int size){

	MatrixXd resultr(size,size);
	MatrixXd resulti(size,size);
	MatrixXcd C1(size,size);
	MatrixXcd C2(size,size);
	complex<double> Im(0,1);
	complex<double> Re(1,0);

    double *rbuff = new double[1000000];
    double *ibuff = new double[1000000];
    string aux;

    ifstream infile;
    infile.open("density.dat");
    for (int i=0; i<size; i++){
		string line;
		getline(infile, line);
		stringstream stream(line);
		for (int j=0; j<size; j++){
			getline(stream, aux, '(');
			getline(stream, aux, ',');
			rbuff[i*size+j] = stod(aux);
			getline(stream, aux, ')');
			ibuff[i*size+j] = stod(aux);

        }
	}

    infile.close();

    // Populate matrix with numbers.
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
			resultr(i,j) = rbuff[i*size+j];
			resulti(i,j) = ibuff[i*size+j];

		}
	}
    C1 = Re * resultr;
    C2 = Im * resulti;
    delete[] rbuff;
    delete[] ibuff;
    return C1+C2;
}

void Fixtbel::write_matrix(MatrixXcd restartdens){
	ofstream file("density.dat");
	IOFormat HeavyFmt(FullPrecision);
    if (file.is_open())
    {
      file << restartdens.format(HeavyFmt);
    }
}

void Fixtbel::applybias(Ref<MatrixXd> fock_ab,
  int *biasedatoms,
  double biasterm,
  int size)
{
  int i;
  for(i=0; i<size; i++)
  {
    fock_ab(i,i) += biasterm * biasedatoms[i];
  }
  return;
}

MatrixXcd Fixtbel::getgs(
  Ref<MatrixXd> fock)
  {
	int nat = atom->nlocal + nreact;
    int i, j, k, coef[nat];
    double val, valt;

    EigenSolver<MatrixXd> fock_mb(fock, true);
    MatrixXd eigv = fock_mb.eigenvalues().real();
    MatrixXcd V = fock_mb.eigenvectors();
    MatrixXcd dens_mb = MatrixXcd::Zero(nat,nat);

    for(i=0; i<nat; i++) coef[i] = 0;
    i=0;
    while(i<nat)
    {
      valt = 100.0;
      k=0;
      for(j=0; j<nat; j++)
      {
        val = eigv(j);
        if(val<valt && coef[j]==0)
        {
          valt = val;
          k=j;
        }
      }
      coef[k] = 1;
      i+=2;
    }
    for(i=0; i<nat; i++) dens_mb(i,i) = coef[i];

    return V * dens_mb * V.inverse();
  }

double Fixtbel::matrixdiff(Ref<MatrixXcd> A, Ref<MatrixXcd> B)
{
	double result = 0.0;
	int nat = atom->nlocal + nreact;
	for(int i=0; i<nat; i++){
		for(int j=0; j<nat; j++){
			result += norm(A(i,j)-B(i,j));
		}
	}
	return result;
}

MatrixXcd Fixtbel::find_gs(Ref<MatrixXd> fock)
{
	MatrixXd fockint = fock;
	MatrixXcd dens = getgs(fock);
	MatrixXcd dens_old = dens;
	double residue = 1;

	while (residue > 0.001){
		fockint = fock + 0.0 * makehubbard(dens_old);
		dens = getgs(fockint);
		dens_old = 0.5 * dens + 0.5 * dens_old;
		residue = matrixdiff(dens,dens_old);
		printf("densdiff = %f \n",residue);
	}
	return dens;
}


MatrixXd Fixtbel::stripdensref()
{
  int i, j;
  int nlocal = atom->nlocal;
  MatrixXd m = MatrixXd::Zero(nlocal+nreact,nlocal+nreact);
  for(j=0; j<nlocal+nreact; j++)
  {
	if(biasedatoms[j]){
    for(i=0; i<nlocal+nreact; i++)
    {
     if(biasedatoms[i]) m(i,j) = 1;
     else{
	   m(i,j) = 0.5;
       m(j,i) = 0.5;
     }
    }
   }
  }
  return m;
}

MatrixXd Fixtbel::makehubbard(Ref<MatrixXcd> dens_ab)
{
  int nlocal = atom->nlocal;

  MatrixXd result = MatrixXd::Zero(nlocal+nreact,nlocal+nreact);
  for(int i=0; i<nlocal+nreact; i++){
	  if(reacte[i] || reactg[i]){
		  result(i,i) = 0.25 - pow((dens_ab(i,i).real() - 0.5), 2);
	  }
  }
  return result;
}

MatrixXcd Fixtbel::rungecuta(Ref<MatrixXcd> dens,
  Ref<MatrixXd> fock,
  Ref<MatrixXd> refshape,
  Ref<MatrixXd> hubbard,
  double tstep,
  double drate,
  Ref<MatrixXcd> densref,
  int step)
{
  complex<double> im(0.0,-1.0*tstep);

  MatrixXcd densdiff = refshape.cwiseProduct(dens - densref);
  hubbard = fock + hubbard;

  MatrixXcd k1 = im * (hubbard * dens - dens * hubbard);
  MatrixXcd k2 = im * (hubbard * (dens+0.5*k1)-(dens+0.5*k1) * hubbard);
  //printf("charge input = %6.5f\n", (drate * densdiff).trace().real());
  return dens + k2 - tstep * drate * densdiff;
}

void Fixtbel::outinitial(Ref<MatrixXd> fock)
{
  ofstream out;
  out.open ("TB_gs.dat");
  out << "# FOCK MATRIX\n";
  out << fock << "\n\n";
  out.close();
  return;
}

void Fixtbel::outstepcharge(Ref<MatrixXd> fock, Ref<MatrixXcd> dens_ab, int step)
{
  double energy;
  int i, iglobal;
  double chargei[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  energy = (fock * dens_ab).trace().real();

  for(i=0; i<nreg; i++)
  {
	iglobal = local_reg_list[i];
    if(biasedatoms[iglobal]) chargei[0] += dens_ab(i,i).real();
    else if(reactg[iglobal]) chargei[1] += dens_ab(i,i).real();
    else if(electrode[iglobal]) chargei[4] += dens_ab(i,i).real();
    else if(prod[iglobal]) chargei[2] += dens_ab(i,i).real();
    else if(surf[iglobal]) chargei[3] += dens_ab(i,i).real();
  }
    for(i=nreg; i<nreg+nlocreact; i++)
  {
	chargei[5] += dens_ab(i,i).real();
  }
  printf("TBstep: %3.0f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f \n",
  step*0.01, energy, chargei[0], chargei[4], chargei[3], chargei[1],
  chargei[5], chargei[2]);
  return;
}

void Fixtbel::outcharge(Ref<MatrixXcd> dens_ab)
{
  double **x = atom->x;
  int i, k = 0;
  int nlocal = atom->nlocal;
  double charge;

  printf("Charges:");
  for(i=0; i<nlocal; i++)
  {
    if(reactg[i]){
		charge = dens_ab(i,i).real() + dens_ab(nlocal+k,nlocal+k).real();
		printf("  %6.4f",charge);
		k++;
	}
  }
  printf("\n");
  return;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */
//Esteban: asegurarse de que actualize correctamente luego de una reaccion

void Fixtbel::update_gas_atoms_list()
{
//printf("Begin of Fixtbel::update_gas_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

  if (nlocal > gcmc_nmax) {
    memory->sfree(local_gas_list);
    gcmc_nmax = atom->nmax;
    local_gas_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_gas_list");
    memory->sfree(local_react_list);
    gcmc_nmax = atom->nmax;
    local_react_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_react_list");
    memory->sfree(local_locreact_list);
    gcmc_nmax = atom->nmax;
    local_locreact_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_locreact_list");
     memory->sfree(local_reg_list);
     gcmc_nmax = atom->nmax;
     local_reg_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
      "GCMC:local_reg_list");
  }

  ngas_local = 0;
 //printf("End of Fixtbel::update_gas_atoms_list()\n");

}

/* ----------------------------------------------------------------------
   update the list of reactive atoms
------------------------------------------------------------------------- */

void Fixtbel::update_reactive_atoms_list()
{
 //if (comm->me == 0) printf("Begin of Fixtbel::update_reactive_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

//    if (nlocal > gcmc_nmax) {
//    printf("Hasta aca\n");
//    memory->sfree(local_react_list);
//    gcmc_nmax = atom->nmax;
//    local_react_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
//     "GCMC:local_react_list");
//  }

  nreact_local = 0;

    int *type = atom->type; // added by Jibao

  if (regionflag) {

      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == reactive_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          //if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_react_list[nreact_local] = i;
            nreact_local++;
          //}
        }
      }


  } else {
    for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (type[i] == reactive_type)) { // Modified by Jibao
        //if (type[i] == reactive_type) {
      //if (mask[i] & groupbit) {   // commented out by Jibao
        local_react_list[nreact_local] = i;
        nreact_local++;
      }
    }
  }

  MPI_Allreduce(&nreact_local,&nreact,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nreact_local,&nreact_before,1,MPI_INT,MPI_SUM,world);
  nreact_before -= nreact_local;
  //printf("proc=%i, nlocal=%i, nreact_local=%i, nreact_before=%i\n", comm->me, nlocal, nreact_local, nreact_before);
}

/* ----------------------------------------------------------------------
   update the list of product atoms
------------------------------------------------------------------------- */

void Fixtbel::update_locreact_atoms_list()
{
 //if (comm->me == 0) printf("Begin of Fixtbel::update_product_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

  nlocreact_local = 0;

    int *type = atom->type;

  if (regionflag) {

 //Esteban: modificado para trabajar con el numero de productos
      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == reactive_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          if (iregion->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_locreact_list[nlocreact_local] = i;
            nlocreact_local++;
          }
        }
      }


  } else {
    for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (type[i] == reactive_type)) { // Modified by Jibao
        //if (type[i] == reactive_type) {
      //if (mask[i] & groupbit) {   // commented out by Jibao
        local_locreact_list[nlocreact_local] = i;
        nlocreact_local++;
      }
    }
  }

  MPI_Allreduce(&nlocreact_local,&nlocreact,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nlocreact_local,&nlocreact_before,1,MPI_INT,MPI_SUM,world);
  nlocreact_before -= nlocreact_local;
  //printf("proc=%i, nlocal=%i, nprod_local=%i, nprod_before=%i\n", comm->me, nlocal, nprod_local, nprod_before);
}

/* ----------------------------------------------------------------------
   update the list of reactive atoms
------------------------------------------------------------------------- */

void Fixtbel::update_region_atoms_list()
{
 //if (comm->me == 0) printf("Begin of Fixtbel::update_reactive_atoms_list()\n");
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *type = atom->type;
  int kin = 0;
  int kout = 0;

  nreg_local = 0;

  if (regionflag) {
    for (int i = 0; i < nlocal; i++) {
        if (iregion->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_reg_list[nreg_local] = i;
            nreg_local++;
            if (type[i] == reactive_type){
				local_reg_list[nlocal+kin] = nlocal+kout;
				kin++;
	    }
	    if (type[i] == reactive_type){
			kout++;
		}
      }
    }

  }

  else {
	  error->all(FLERR,"must have region");
  }

  MPI_Allreduce(&nreg_local,&nreg,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nreg_local,&nreg_before,1,MPI_INT,MPI_SUM,world);
  nreg_before -= nreg_local;
  //printf("proc=%i, nlocal=%i, nreact_local=%i, nreact_before=%i\n", comm->me, nlocal, nreact_local, nreact_before);
}

/* ----------------------------------------------------------------------
  return acceptance ratios
------------------------------------------------------------------------- */

double Fixtbel::compute_vector(int n)
{
  //if (n == 0) return ntranslation_attempts;

  return 0.0;
}
//Esteban:Agregar los nuevos eventos

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double Fixtbel::memory_usage()
{
  double bytes = gcmc_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void Fixtbel::write_restart(FILE *fp)
{
  int n = 0;
  double list[4];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = next_reneighbor;
  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void Fixtbel::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = static_cast<int> (list[n++]);
}
