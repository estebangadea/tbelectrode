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
#include "fix_gckmc_new.h"
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
#include "pair_hybrid_overlay.h"    // added by Jibao
#include "pair_sw.h"        // added by Jibao
//#include "pair_sw0.h"        // added by Jibao
#include "pair_hybrid.h"        // added by Jibao

using namespace std;
using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};

/* ---------------------------------------------------------------------- */

FixGCkMC::FixGCkMC(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  //printf("Beginin of FixGCkMC\n");
  if (narg < 10) error->all(FLERR,"Esteban: Illegal fix gcmc command");

  if (atom->molecular == 2)
    error->all(FLERR,"Fix gcmc does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  //size_vector = 8; // commented out by Jibao
    size_vector = 13;    // added by Jibao according to Matias' 2012 lammps version, to output energyout
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // required args


  reservoir_temperature = force->numeric(FLERR,arg[3]);
  reactive_type = force->inumeric(FLERR,arg[4]);
  product_type = force->inumeric(FLERR,arg[5]);
  surf_type = force->inumeric(FLERR,arg[6]);
  preexp = force->numeric(FLERR,arg[7]);
  potential = force->numeric(FLERR,arg[8]);
  seed = force->inumeric(FLERR,arg[9]);

 //Esteban: reactive_type, product_type, E, region, nreactions

  if (seed <= 0) error->all(FLERR,"Illegal fix gcmc command");
  if (reservoir_temperature < 0.0)
    error->all(FLERR,"Illegal fix gcmc command");

    //molflag = 0; // variable in 2012 verion // Jibao
    pairflag = 0; // added by Jibao. from Matias
    //pressflag=0;    // added by Jibao. from Matias
    regionflag=0;   // added by Jibao. from Matias

  // read options from end of input line

  options(narg-10,&arg[10]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // error checks on region and its extent being inside simulation box

  region_xlo = region_xhi = region_ylo = region_yhi =
    region_zlo = region_zhi = 0.0;
  if (regionflag) {
    if (domain->regions[iregion]->bboxflag == 0)
      error->all(FLERR,"Fix gcmc region does not support a bounding box");
    if (domain->regions[iregion]->dynamic_check())
      error->all(FLERR,"Fix gcmc region cannot be dynamic");

    region_xlo = domain->regions[iregion]->extent_xlo;
    region_xhi = domain->regions[iregion]->extent_xhi;
    region_ylo = domain->regions[iregion]->extent_ylo;
    region_yhi = domain->regions[iregion]->extent_yhi;
    region_zlo = domain->regions[iregion]->extent_zlo;
    region_zhi = domain->regions[iregion]->extent_zhi;

    if (region_xlo < domain->boxlo[0] || region_xhi > domain->boxhi[0] ||
        region_ylo < domain->boxlo[1] || region_yhi > domain->boxhi[1] ||
        region_zlo < domain->boxlo[2] || region_zhi > domain->boxhi[2])
      error->all(FLERR,"Fix gcmc region extends outside simulation box");

    // estimate region volume using MC trials

    double coord[3];
    int inside = 0;
    int attempts = 10000000;
    for (int i = 0; i < attempts; i++) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      if (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) != 0)
        inside++;
    }

    double max_region_volume = (region_xhi - region_xlo)*
     (region_yhi - region_ylo)*(region_zhi - region_zlo);

    region_volume = max_region_volume*static_cast<double> (inside)/
     static_cast<double> (attempts);
  }

  // error check and further setup for mode = MOLECULE

  if (mode == MOLECULE) {
    if (onemols[imol]->xflag == 0)
      error->all(FLERR,"Fix gcmc molecule must have coordinates");
    if (onemols[imol]->typeflag == 0)
      error->all(FLERR,"Fix gcmc molecule must have atom types");
    if (ngcmc_type != 0)
      error->all(FLERR,"Atom type must be zero in fix gcmc mol command");
    if (onemols[imol]->qflag == 1 && atom->q == NULL)
      error->all(FLERR,"Fix gcmc molecule has charges, but atom style does not");

    if (atom->molecular == 2 && onemols != atom->avec->onemols)
      error->all(FLERR,"Fix gcmc molecule template ID must be same "
                 "as atom_style template ID");
    onemols[imol]->check_attributes(0);
  }

  if (charge_flag && atom->q == NULL)
    error->all(FLERR,"Fix gcmc atom has charge, but atom style does not");

  if (shakeflag && mode == ATOM)
    error->all(FLERR,"Cannot use fix gcmc shake and not molecule");

  // setup of coords and imageflags array

  if (mode == ATOM) natoms_per_molecule = 1;
  else natoms_per_molecule = onemols[imol]->natoms;
  memory->create(coords,natoms_per_molecule,3,"gcmc:coords");
  memory->create(imageflags,natoms_per_molecule,"gcmc:imageflags");
  memory->create(atom_coord,natoms_per_molecule,3,"gcmc:atom_coord");

  // compute the number of MC cycles that occur nevery timesteps

  //ncycles = nexchanges + nmcmoves + nreactions; //Esteban: Agregar nreactions

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  ntranslation_attempts = 0.0;
  ntranslation_successes = 0.0;
  nrotation_attempts = 0.0;
  nrotation_successes = 0.0;
  ndeletion_attempts = 0.0;
  ndeletion_successes = 0.0;
  ninsertion_attempts = 0.0;
  ninsertion_successes = 0.0;
  nfreaction_attempts = 0.0;
  nfreaction_successes = 0.0;
  nbreaction_attempts = 0.0;
  nbreaction_successes = 0.0;


  //Esteban: nfreaction_attempts, nbreaction_attempts, nfreaction_successes, nbreaction_successes

    energyout=0.0;  // Matias

  gcmc_nmax = 0;
  local_gas_list = NULL;
  local_react_list = NULL; //Esteban
  local_locreact_list = NULL;
  local_reg_list = NULL;  //Esteban
  biasedatoms = NULL;
  electrode = NULL;
  surf = NULL;
  reacte = NULL;
  reactg = NULL;
  prod = NULL;
  

    if (comm->me == 0) printf("End of FixGCkMC::FixGCkMC()\n");
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixGCkMC::options(int narg, char **arg)
{
  //printf("Begin_option\n");
  if (narg < 0) error->all(FLERR,"Illegal fix gcmc command");

  // defaults

  mode = ATOM;
  max_rotation_angle = 10*MY_PI/180;
  regionflag = 0;
  iregion = -1;
  region_volume = 0;
  max_region_attempts = 1000;
  molecule_group = 0;
  molecule_group_bit = 0;
  molecule_group_inversebit = 0;
  exclusion_group = 0;
  exclusion_group_bit = 0;
  pressure_flag = false;
  pressure = 0.0;
  fugacity_coeff = 1.0;
  shakeflag = 0;
  charge = 0.0;
  charge_flag = false;
  full_flag = false;
  idshake = NULL;
  ngroups = 0;
  int ngroupsmax = 0;
  groupstrings = NULL;
  ngrouptypes = 0;
  int ngrouptypesmax = 0;
  grouptypestrings = NULL;
  grouptypes = NULL;
  grouptypebits = NULL;
  energy_intra = 0.0;
  tfac_insert = 1.0;

  int iarg = 0;
  while (iarg < narg) {
  if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1)
        error->all(FLERR,"Molecule template ID for fix gcmc does not exist");
      if (atom->molecules[imol]->nset > 1 && comm->me == 0)
        error->warning(FLERR,"Molecule template for "
                       "fix gcmc has multiple molecules");
      mode = MOLECULE;
      onemols = atom->molecules;
      nmol = onemols[imol]->nset;
      iarg += 2;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix gcmc does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"maxangle") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      max_rotation_angle = force->numeric(FLERR,arg[iarg+1]);
      max_rotation_angle *= MY_PI/180;
      iarg += 2;
    } else if (strcmp(arg[iarg],"pair") == 0) { // added by Jibao. from Matias
        if (iarg+2 > narg) error->all(FLERR,"Illegal fix GCMC command");
        if (strcmp(arg[iarg+1],"lj/cut") == 0) pairflag = 0;
        else if (strcmp(arg[iarg+1],"Stw") == 0) pairflag = 1;
        else error->all(FLERR,"Illegal fix evaporate command");
        iarg += 2;
    }   // Matias
    else if (strcmp(arg[iarg],"pressure") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      pressure = force->numeric(FLERR,arg[iarg+1]);
        pressure = pressure * 100.0;    // added by Jibao, according to Matias' code
      pressure_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"fugacity_coeff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      fugacity_coeff = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"charge") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      charge = force->numeric(FLERR,arg[iarg+1]);
      charge_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"shake") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      int n = strlen(arg[iarg+1]) + 1;
      delete [] idshake;
      idshake = new char[n];
      strcpy(idshake,arg[iarg+1]);
      shakeflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"full_energy") == 0) {
      full_flag = true;
      iarg += 1;
    } else if (strcmp(arg[iarg],"group") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      if (ngroups >= ngroupsmax) {
	ngroupsmax = ngroups+1;
	groupstrings = (char **)
	  memory->srealloc(groupstrings,
			   ngroupsmax*sizeof(char *),
			   "fix_gcmc:groupstrings");
      }
      int n = strlen(arg[iarg+1]) + 1;
      groupstrings[ngroups] = new char[n];
      strcpy(groupstrings[ngroups],arg[iarg+1]);
      ngroups++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"grouptype") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix gcmc command");
      if (ngrouptypes >= ngrouptypesmax) {
	ngrouptypesmax = ngrouptypes+1;
	grouptypes = (int*) memory->srealloc(grouptypes,ngrouptypesmax*sizeof(int),
			 "fix_gcmc:grouptypes");
	grouptypestrings = (char**)
	  memory->srealloc(grouptypestrings,
			   ngrouptypesmax*sizeof(char *),
			   "fix_gcmc:grouptypestrings");
      }
      grouptypes[ngrouptypes] = atoi(arg[iarg+1]);
      int n = strlen(arg[iarg+2]) + 1;
      grouptypestrings[ngrouptypes] = new char[n];
      strcpy(grouptypestrings[ngrouptypes],arg[iarg+2]);
      ngrouptypes++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"intra_energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      energy_intra = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tfac_insert") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      tfac_insert = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix gcmc command");
  }
    //  if (comm->me == 0) printf("End of FixGCkMC::options()\n");
}

/* ---------------------------------------------------------------------- */

FixGCkMC::~FixGCkMC()
{
  //  printf("FixGCkMC()");
  if (regionflag) delete [] idregion;
  delete random_equal;
  delete random_unequal;

    //delete region_insert;   // from Matias; deleted by Jibao

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

  delete [] idshake;

  if (ngroups > 0) {
    for (int igroup = 0; igroup < ngroups; igroup++)
      delete [] groupstrings[igroup];
    memory->sfree(groupstrings);
  }

  if (ngrouptypes > 0) {
    memory->destroy(grouptypes);
    memory->destroy(grouptypebits);
    for (int igroup = 0; igroup < ngrouptypes; igroup++)
      delete [] grouptypestrings[igroup];
    memory->sfree(grouptypestrings);
  }
   // if (comm->me == 0) printf("End of FixGCkMC::~FixGCkMC()\n");
}

/* ---------------------------------------------------------------------- */

int FixGCkMC::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGCkMC::init()
{
  //  if (comm->me == 0) printf("Begins: FixGCkMC::init()\n");

  triclinic = domain->triclinic;

  // decide whether to switch to the full_energy option

  if (!full_flag) {
    if ((force->kspace) ||
        (force->pair == NULL) ||
        (force->pair->single_enable == 0) ||
        (force->pair_match("hybrid",0)) ||
        (force->pair_match("eam",0))
	) {
      full_flag = true;

        //if (comm->me == 0) printf("Begins: inside if (!full_flag){}: FixGCkMC::init()\n");

        //if (comm->me == 0) printf("pairflag = %d\n",pairflag);

        if (pairflag) { // added by Jibao
            full_flag = false;  // added by Jibao
        }   // added by Jibao

      if (comm->me == 0 && full_flag == true) // modified by Jibao
          error->warning(FLERR,"Fix gcmc using full_energy option");
    }
  }

    //if (comm->me == 0) printf("Begins 2: FixGCkMC::init()\n");

  if (full_flag) {
    char *id_pe = (char *) "thermo_pe";
    int ipe = modify->find_compute(id_pe);
    c_pe = modify->compute[ipe];
  }

  int *type = atom->type;

  if (mode == ATOM) {
    if (product_type <= 0 || product_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix gcmc command");
    if (reactive_type <= 0 || reactive_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix gcmc command");
  }

    //if (comm->me == 0) printf("Begins 3: FixGCkMC::init()\n");

  // if mode == ATOM, warn if any deletable atom has a mol ID

  if ((mode == ATOM) && atom->molecule_flag) {
      /*
      if (comm->me == 0) {
          printf("Inside if ((mode == ATOM)): FixGCkMC::init()\n");
          printf("atom->molecule_flag = %d\n",atom->molecule_flag);
      }
      */
    tagint *molecule = atom->molecule;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (type[i] == reactive_type)
        if (molecule[i]) flag = 1;

      //if (comm->me == 0) printf("Inside if ((mode == ATOM)) 2: FixGCkMC::init()\n");

    int flagall;

      //printf("comm->me = %d, flag = %d, flagall = %d, before MPI_ALLreduce()\n",comm->me,flag,flagall);

      //error->all(FLERR,"Kao 0 !!!!");    // added by Jibao

    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

      //error->all(FLERR,"Kao 1 !!!!");    // added by Jibao

      //if (comm->me == 0) printf("Inside if ((mode == ATOM)) 3: FixGCkMC::init()\n");
      //if (comm->me == 0) printf("flag = %d, flagall = %d, after MPI_ALLreduce()\n",flag,flagall);

      if (flagall && comm->me == 0) {
          //if (comm->me == 0) printf("Inside if if (flagall && comm->me == 0): FixGCkMC::init()\n");    // added by Jibao
          //error->all(FLERR,"Kao 2 !!!!");    // added by Jibao
          error->all(FLERR,"Fix gcmc cannot exchange individual atoms belonging to a molecule");
      }
  }

    //if (comm->me == 0) printf("Begins 4: FixGCkMC::init()\n");

  // if mode == MOLECULE, check for unset mol IDs

  if (mode == MOLECULE) {
    tagint *molecule = atom->molecule;
    int *mask = atom->mask;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (mask[i] == groupbit)
        if (molecule[i] == 0) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
    if (flagall && comm->me == 0)
      error->all(FLERR,
       "All mol IDs should be set for fix gcmc group atoms");
  }

    //if (comm->me == 0) printf("Begins 5: FixGCkMC::init()\n");

  if (((mode == MOLECULE) && (atom->molecule_flag == 0)) ||
      ((mode == MOLECULE) && (!atom->tag_enable || !atom->map_style)))
    error->all(FLERR,
               "Fix gcmc molecule command requires that "
               "atoms have molecule attributes");

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one

    //if (comm->me == 0) printf("Begins 6: FixGCkMC::init()\n");

  fixshake = NULL;
  if (shakeflag) {
    int ifix = modify->find_fix(idshake);
    if (ifix < 0) error->all(FLERR,"Fix gcmc shake fix does not exist");
    fixshake = modify->fix[ifix];
    int tmp;
    if (onemols != (Molecule **) fixshake->extract("onemol",tmp))
      error->all(FLERR,"Fix gcmc and fix shake not using "
                 "same molecule template ID");
  }

    //if (comm->me == 0) printf("Begins 7: FixGCkMC::init()\n");

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix gcmc in a 2d simulation");

  // create a new group for interaction exclusions

    //if (comm->me == 0) printf("Before 'create a new group for interaction exclusions': FixGCkMC::init()\n");

  if (full_flag || pairflag) {  // modified by Jibao; added "|| pairflag"
    char **group_arg = new char*[4];
    // create unique group name for atoms to be excluded
    int len = strlen(id) + 30;
    group_arg[0] = new char[len];
    sprintf(group_arg[0],"FixGCMC:gcmc_exclusion_group:%s",id);
    group_arg[1] = (char *) "subtract";
    group_arg[2] = (char *) "all";
    group_arg[3] = (char *) "all";
    group->assign(4,group_arg);
    exclusion_group = group->find(group_arg[0]);
    if (exclusion_group == -1)
      error->all(FLERR,"Could not find fix gcmc exclusion group ID");
    exclusion_group_bit = group->bitmask[exclusion_group];

    // neighbor list exclusion setup
    // turn off interactions between group all and the exclusion group

    int narg = 4;
    char **arg = new char*[narg];;
    arg[0] = (char *) "exclude";
    arg[1] = (char *) "group";
    arg[2] = group_arg[0];
    arg[3] = (char *) "all";
    neighbor->modify_params(narg,arg);
    delete [] group_arg[0];
    delete [] group_arg;
    delete [] arg;
  }

  // create a new group for temporary use with selected molecules

  if (mode == MOLECULE) {
    char **group_arg = new char*[3];
    // create unique group name for atoms to be rotated
    int len = strlen(id) + 30;
    group_arg[0] = new char[len];
    sprintf(group_arg[0],"FixGCMC:rotation_gas_atoms:%s",id);
    group_arg[1] = (char *) "molecule";
    char digits[12];
    sprintf(digits,"%d",-1);
    group_arg[2] = digits;
    group->assign(3,group_arg);
    molecule_group = group->find(group_arg[0]);
    if (molecule_group == -1)
      error->all(FLERR,"Could not find fix gcmc rotation group ID");
    molecule_group_bit = group->bitmask[molecule_group];
    molecule_group_inversebit = molecule_group_bit ^ ~0;
    delete [] group_arg[0];
    delete [] group_arg;
  }

  // get all of the needed molecule data if mode == MOLECULE,
  // otherwise just get the gas mass

  if (mode == MOLECULE) {

    onemols[imol]->compute_mass();
    onemols[imol]->compute_com();
    gas_mass = onemols[imol]->masstotal;
    for (int i = 0; i < onemols[imol]->natoms; i++) {
      onemols[imol]->x[i][0] -= onemols[imol]->com[0];
      onemols[imol]->x[i][1] -= onemols[imol]->com[1];
      onemols[imol]->x[i][2] -= onemols[imol]->com[2];
    }

  } else //gas_mass = atom->mass[ngcmc_type];

  //if (gas_mass <= 0.0)
  //  error->all(FLERR,"Illegal fix gcmc gas mass <= 0");

  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do GCMC on atoms in atom_modify first group");
  }

  beta = 1.0/(1.38065e-23*reservoir_temperature);

  kfreact = preexp*exp(0.5*1*1.6022e-19*beta*potential); //Esteban: Potencial estandard de la oxidacion del agua en V.
  kbreact = preexp*exp(-0.5*1*1.6022e-19*beta*potential);

  imagezero = ((imageint) IMGMAX << IMG2BITS) |
             ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // construct group bitmask for all new atoms
  // aggregated over all group keywords

  groupbitall = 1 | groupbit;
  for (int igroup = 0; igroup < ngroups; igroup++) {
    int jgroup = group->find(groupstrings[igroup]);
    if (jgroup == -1)
      error->all(FLERR,"Could not find specified fix gcmc group ID");
    groupbitall |= group->bitmask[jgroup];
  }

  // construct group type bitmasks
  // not aggregated over all group keywords

  if (ngrouptypes > 0) {
    memory->create(grouptypebits,ngrouptypes,"fix_gcmc:grouptypebits");
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      int jgroup = group->find(grouptypestrings[igroup]);
      if (jgroup == -1)
	error->all(FLERR,"Could not find specified fix gcmc group ID");
      grouptypebits[igroup] = group->bitmask[jgroup];
    }
  }

  //  printf("End of FixGCkMC::init()\n");

}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixGCkMC::pre_exchange()
{
  // just return if should not be called on this timestep
 //if (comm->me == 0) printf("Begin of FixGCkMC::pre_exchange()\n");
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

  if (regionflag) volume = region_volume;
  else volume = domain->xprd * domain->yprd * domain->zprd;

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

  if (full_flag) {
    error->all(FLERR,"gcmc/react does not allow full energy");

  } else {

    if (mode == MOLECULE) {
      error->all(FLERR,"gcmc/react does not allow mode MOLECULE");
    } else {
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
        
        
        
        for(int ii=0; ii<8268; ii++)
        {
		hubbard = makehubbard(dens_ab);
		local_hubbard = 0.1 * localizeMatrixd(hubbard);
        local_dens_ab = rungecuta(local_dens_ab,
          local_fock_ab,
          local_refshape,
          local_hubbard,
          0.0050, //tstep
          0.002, //driving rate
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

     }
            //Esteban: Agregar la probabilidad de una freaction o breaction

      //domain->pbc();    // added by Jibao; to prevent the error: ERROR on proc 0: Bond atoms 4205 4209 missing on proc 0 at step 285 (../neigh_bond.cpp:196)
      //comm->exchange(); // added by Jibao; to prevent the error: ERROR on proc 0: Bond atoms 4205 4209 missing on proc 0 at step 285 (../neigh_bond.cpp:196)
  }
  
  next_reneighbor = update->ntimestep+1;
 //if (comm->me == 0) printf("End of FixGCkMC::pre_exchange()\n");
}

void FixGCkMC::fill_lists(int *biasedatoms,
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
MatrixXd FixGCkMC::localizeMatrixd(Ref<MatrixXd> m){
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

MatrixXcd FixGCkMC::localizeMatrix(Ref<MatrixXcd> m){
	
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

void FixGCkMC::update_density(Ref<MatrixXcd> dens_ab, Ref<MatrixXcd> local_dens_ab){
	
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


void FixGCkMC::fill_fock(Ref<MatrixXd> fock_ab){
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

bool FixGCkMC::exists_test (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

MatrixXcd FixGCkMC::readmatrix(int size){
	
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
    
void FixGCkMC::write_matrix(MatrixXcd restartdens){
	ofstream file("density.dat");
	IOFormat HeavyFmt(FullPrecision);
    if (file.is_open())
    {
      file << restartdens.format(HeavyFmt);
    }
}

void FixGCkMC::applybias(Ref<MatrixXd> fock_ab,
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

MatrixXcd FixGCkMC::getgs(
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

double FixGCkMC::matrixdiff(Ref<MatrixXcd> A, Ref<MatrixXcd> B)
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
  
MatrixXcd FixGCkMC::find_gs(Ref<MatrixXd> fock)
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


MatrixXd FixGCkMC::stripdensref()
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

MatrixXd FixGCkMC::makehubbard(Ref<MatrixXcd> dens_ab)
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

MatrixXcd FixGCkMC::rungecuta(Ref<MatrixXcd> dens,
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

void FixGCkMC::outinitial(Ref<MatrixXd> fock)
{
  ofstream out;
  out.open ("TB_gs.dat");
  out << "# FOCK MATRIX\n";
  out << fock << "\n\n";
  out.close();
  return;
}

void FixGCkMC::outstepcharge(Ref<MatrixXd> fock, Ref<MatrixXcd> dens_ab, int step)
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

void FixGCkMC::outcharge(Ref<MatrixXcd> dens_ab)
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
------------------------------------------------------------------------- */

void FixGCkMC::toggle_intramolecular(int i)
{
  if (atom->avec->bonds_allow)
    for (int m = 0; m < atom->num_bond[i]; m++)
      atom->bond_type[i][m] = -atom->bond_type[i][m];

  if (atom->avec->angles_allow)
    for (int m = 0; m < atom->num_angle[i]; m++)
      atom->angle_type[i][m] = -atom->angle_type[i][m];

  if (atom->avec->dihedrals_allow)
    for (int m = 0; m < atom->num_dihedral[i]; m++)
      atom->dihedral_type[i][m] = -atom->dihedral_type[i][m];

  if (atom->avec->impropers_allow)
    for (int m = 0; m < atom->num_improper[i]; m++)
      atom->improper_type[i][m] = -atom->improper_type[i][m];
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */
//Esteban: asegurarse de que actualize correctamente luego de una reaccion

void FixGCkMC::update_gas_atoms_list()
{
//printf("Begin of FixGCkMC::update_gas_atoms_list()\n");
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
 //printf("End of FixGCkMC::update_gas_atoms_list()\n");

}

/* ----------------------------------------------------------------------
   update the list of reactive atoms
------------------------------------------------------------------------- */

void FixGCkMC::update_reactive_atoms_list()
{
 //if (comm->me == 0) printf("Begin of FixGCkMC::update_reactive_atoms_list()\n");
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

    if (mode == MOLECULE) {

      tagint maxmol = 0;
      for (int i = 0; i < nlocal; i++) maxmol = MAX(maxmol,molecule[i]);
      tagint maxmol_all;
      MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
      double comx[maxmol_all];
      double comy[maxmol_all];
      double comz[maxmol_all];
      for (int imolecule = 0; imolecule < maxmol_all; imolecule++) {
        for (int i = 0; i < nlocal; i++) {
          if (molecule[i] == imolecule) {
            mask[i] |= molecule_group_bit;
          } else {
            mask[i] &= molecule_group_inversebit;
          }
        }
        double com[3];
        com[0] = com[1] = com[2] = 0.0;
        group->xcm(molecule_group,gas_mass,com);
        comx[imolecule] = com[0];
        comy[imolecule] = com[1];
        comz[imolecule] = com[2];
      }

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          if (domain->regions[iregion]->match(comx[molecule[i]],
             comy[molecule[i]],comz[molecule[i]]) == 1) {
            local_gas_list[ngas_local] = i;
            ngas_local++;
          }
        }
      }

    } else { //Esteban: modificado para trabajar con el numero de reactivos
      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == reactive_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          //if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_react_list[nreact_local] = i;
            nreact_local++;
          //}
        }
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

void FixGCkMC::update_locreact_atoms_list()
{
 //if (comm->me == 0) printf("Begin of FixGCkMC::update_product_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

//  if (nlocal > gcmc_nmax) {
//    memory->sfree(local_prod_list);
//    gcmc_nmax = atom->nmax;
//    local_prod_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
//     "GCMC:local_prod_list");
//  }

  nlocreact_local = 0;

    int *type = atom->type; // added by Jibao

  if (regionflag) {

    if (mode == MOLECULE) {

      tagint maxmol = 0;
      for (int i = 0; i < nlocal; i++) maxmol = MAX(maxmol,molecule[i]);
      tagint maxmol_all;
      MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
      double comx[maxmol_all];
      double comy[maxmol_all];
      double comz[maxmol_all];
      for (int imolecule = 0; imolecule < maxmol_all; imolecule++) {
        for (int i = 0; i < nlocal; i++) {
          if (molecule[i] == imolecule) {
            mask[i] |= molecule_group_bit;
          } else {
            mask[i] &= molecule_group_inversebit;
          }
        }
        double com[3];
        com[0] = com[1] = com[2] = 0.0;
        group->xcm(molecule_group,gas_mass,com);
        comx[imolecule] = com[0];
        comy[imolecule] = com[1];
        comz[imolecule] = com[2];
      }

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          if (domain->regions[iregion]->match(comx[molecule[i]],
             comy[molecule[i]],comz[molecule[i]]) == 1) {
            local_gas_list[ngas_local] = i;
            ngas_local++;
          }
        }
      }

    } else { //Esteban: modificado para trabajar con el numero de productos
      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == reactive_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_locreact_list[nlocreact_local] = i;
            nlocreact_local++;
          }
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

void FixGCkMC::update_region_atoms_list()
{
 //if (comm->me == 0) printf("Begin of FixGCkMC::update_reactive_atoms_list()\n");
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *type = atom->type;
  int kin = 0;
  int kout = 0;

  nreg_local = 0;

  if (regionflag) {
    for (int i = 0; i < nlocal; i++) {
        if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
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

double FixGCkMC::compute_vector(int n)
{
  if (n == 0) return ntranslation_attempts;
  if (n == 1) return ntranslation_successes;
  if (n == 2) return ninsertion_attempts;
  if (n == 3) return ninsertion_successes;
  if (n == 4) return ndeletion_attempts;
  if (n == 5) return ndeletion_successes;
  if (n == 6) return nrotation_attempts;
  if (n == 7) return nrotation_successes;
    if (n == 8) return energyout;   // added by Jibao
  if (n == 9) return nfreaction_attempts;
  if (n == 10) return nfreaction_successes;
  if (n == 11) return nbreaction_attempts;
  if (n == 12) return nbreaction_successes;

  return 0.0;
}
//Esteban:Agregar los nuevos eventos

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGCkMC::memory_usage()
{
  double bytes = gcmc_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixGCkMC::write_restart(FILE *fp)
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

void FixGCkMC::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = static_cast<int> (list[n++]);
}
