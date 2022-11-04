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

#ifdef FIX_CLASS

FixStyle(tbel,FixTBel)

#else

#ifndef LMP_FIX_TBEL_H
#define LMP_FIX_TBEL_H

#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>
#include "fix.h"
//#include "pointers.h"   // added by Jibao, according to Matias' 2012 version


namespace LAMMPS_NS {

class FixTBel : public Fix {
 public:
  FixTBel(class LAMMPS *, int, char **);
  ~FixTBel();
  int setmask();
  void init();
  void pre_exchange();
  void attempt_atomic_translation();
  void attempt_atomic_deletion();
  void attempt_atomic_insertion();
  void attempt_molecule_translation();
  void attempt_molecule_rotation();
  void attempt_molecule_deletion();
  void attempt_molecule_insertion();
  void attempt_atomic_translation_full();
  void attempt_atomic_deletion_full();
  void attempt_atomic_insertion_full();
  void attempt_molecule_translation_full();
  void attempt_molecule_rotation_full();
  void attempt_molecule_deletion_full();
  void attempt_molecule_insertion_full();

  void attempt_atomic_freaction(int, double *); //Esteban
  void attempt_atomic_breaction(int, double *); //Esteban
  void fill_lists(int *, int *, int *, int *, int *, int *);
  void fill_fock(Eigen::Ref<Eigen::MatrixXd>);
  void outinitial(Eigen::Ref<Eigen::MatrixXd>);
  void applybias(Eigen::Ref<Eigen::MatrixXd> fock_ab,
  int *biasedatoms,
  double biasterm,
  int size);
  bool exists_test(const std::string& name);
  Eigen::MatrixXcd readmatrix(int size);
  void write_matrix(Eigen::MatrixXcd restartdens);

  void update_density(Eigen::Ref<Eigen::MatrixXcd> dens_ab,
  Eigen::Ref<Eigen::MatrixXcd> local_dens_ab);

  Eigen::MatrixXd makehubbard(Eigen::Ref<Eigen::MatrixXcd> dens_ab);
  double matrixdiff(Eigen::Ref<Eigen::MatrixXcd> A,
  Eigen::Ref<Eigen::MatrixXcd> B);

  Eigen::MatrixXcd rungecuta(Eigen::Ref<Eigen::MatrixXcd> dens,
  Eigen::Ref<Eigen::MatrixXd> fock,
  Eigen::Ref<Eigen::MatrixXd>refshape,
  Eigen::Ref<Eigen::MatrixXd>hubbard,
  double tstep,
  double drate,
  Eigen::Ref<Eigen::MatrixXcd> densref,
  int step);

  Eigen::MatrixXcd localizeMatrix(Eigen::Ref<Eigen::MatrixXcd> m);
  Eigen::MatrixXd localizeMatrixd(Eigen::Ref<Eigen::MatrixXd> m);

  void outstepcharge(Eigen::Ref<Eigen::MatrixXd> fock,
  Eigen::Ref<Eigen::MatrixXcd> dens_ab,
  int step);
  void outcharge(Eigen::Ref<Eigen::MatrixXcd> dens_ab);

  Eigen::MatrixXd stripdensref();

  Eigen::MatrixXcd getgs(Eigen::Ref<Eigen::MatrixXd> fock);
  Eigen::MatrixXcd find_gs(Eigen::Ref<Eigen::MatrixXd> fock);



  double energy(int, int, tagint, double *);
  double molecule_energy(tagint);
  double energy_full();
  int pick_random_gas_atom();
  int pick_random_reactive_atom(); //Esteban
  int pick_random_product_atom(); //Esteban
  tagint pick_random_gas_molecule();
  void toggle_intramolecular(int);

  void update_gas_atoms_list();
  void update_reactive_atoms_list();
  void update_locreact_atoms_list();
  void update_region_atoms_list();

  double compute_vector(int);
  double memory_usage();
  void write_restart(FILE *);
  void restart(char *);

    void create_gaslist(); // from Matias' version; added by Jibao

 private:


  int exclusion_group,exclusion_group_bit;
  int seed;
  int reactive_type, product_type, surf_type;  //Esteban
  int nreactions; //Esteban
  int ngas;                 // # of gas atoms on all procs
  int nreact, nlocreact, nreg; //Esteban
  int ngas_local;           // # of gas atoms on this proc
  int nreact_local, nlocreact_local, nreg_local; //Esteban
  int ngas_before;          // # of gas atoms on procs < this proc
  int nreact_before, nlocreact_before, nreg_before; //Esteban
  int mode;                 // ATOM or MOLECULE
  int regionflag;           // 0 = anywhere in box, 1 = specific region
  class Region *iregion;            // gcmc region
  char *idregion;           // gcmc region id

  int groupbitall;          // group bitmask for inserted atoms
  int ngroups;              // number of group-ids for inserted atoms
  char** groupstrings;      // list of group-ids for inserted atoms
  int ngrouptypes;          // number of type-based group-ids for inserted atoms
  char** grouptypestrings;  // list of type-based group-ids for inserted atoms
  int* grouptypebits;       // list of type-based group bitmasks
  int* grouptypes;          // list of type-based group types

  double nfreaction_attempts; //Esteban
  double nfreaction_successes;
  double nbreaction_attempts;
  double nbreaction_successes;

  int gcmc_nmax, tbsize;
  int qsteps;
  double reservoir_temperature;
  double volume;
  double potential, qtstep, hubbardp, drate;//Esteban
  double xlo,xhi,ylo,yhi,zlo,zhi;
  double region_xlo,region_xhi,region_ylo,region_yhi,region_zlo,region_zhi;
  double region_volume;
  double *sublo,*subhi;
  int *local_gas_list;
  int *local_react_list;
  int *local_locreact_list;
  int *local_reg_list;
  double **cutsq;
  double **atom_coord;
  int *biasedatoms;
  int *surf;
  int *electrode;
  int *reactg;
  int *reacte;
  int *prod;

  imageint imagezero;

  class Pair *pair;

  class RanPark *random_equal;
  class RanPark *random_unequal;

    class Pair *pairsw; //nuevo; from from Matias' version; added by Jibao; added by Jibao

  class Atom *model_atom;

  int imol,nmol;
  double **coords;
  imageint *imageflags;
  int triclinic;                         // 0 = orthog box, 1 = triclinic

  class Compute *c_pe;

  void options(int, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix gcmc does not (yet) work with atom_style template

Self-explanatory.

E: Fix gcmc region does not support a bounding box
Not all regions represent bounded volumes.  You cannot use
such a region with the fix gcmc command.

E: Fix gcmc region cannot be dynamic

Only static regions can be used with fix gcmc.

E: Fix gcmc region extends outside simulation box

Self-explanatory.

E: Fix gcmc molecule must have coordinates

The defined molecule does not specify coordinates.

E: Fix gcmc molecule must have atom types

The defined molecule does not specify atom types.

E: Atom type must be zero in fix gcmc mol command

Self-explanatory.

E: Fix gcmc molecule has charges, but atom style does not

Self-explanatory.

E: Fix gcmc molecule template ID must be same as atom_style template ID

When using atom_style template, you cannot insert molecules that are
not in that template.

E: Fix gcmc atom has charge, but atom style does not

Self-explanatory.

E: Cannot use fix gcmc shake and not molecule

Self-explanatory.

E: Molecule template ID for fix gcmc does not exist

Self-explanatory.

W: Molecule template for fix gcmc has multiple molecules

The fix gcmc command will only create molecules of a single type,
i.e. the first molecule in the template.

E: Region ID for fix gcmc does not exist

Self-explanatory.

W: Fix gcmc using full_energy option

Fix gcmc has automatically turned on the full_energy option since it
is required for systems like the one specified by the user. User input
included one or more of the following: kspace, triclinic, a hybrid
pair style, an eam pair style, or no "single" function for the pair
style.

E: Invalid atom type in fix gcmc command

The atom type specified in the gcmc command does not exist.

E: Fix gcmc cannot exchange individual atoms belonging to a molecule

This is an error since you should not delete only one atom of a
molecule.  The user has specified atomic (non-molecular) gas
exchanges, but an atom belonging to a molecule could be deleted.

E: All mol IDs should be set for fix gcmc group atoms

The molecule flag is on, yet not all molecule ids in the fix group
have been set to non-zero positive values by the user. This is an
error since all atoms in the fix gcmc group are eligible for deletion,
rotation, and translation and therefore must have valid molecule ids.

E: Fix gcmc molecule command requires that atoms have molecule attributes

Should not choose the gcmc molecule feature if no molecules are being
simulated. The general molecule flag is off, but gcmc's molecule flag
is on.

E: Fix gcmc shake fix does not exist

Self-explanatory.

E: Fix gcmc and fix shake not using same molecule template ID

Self-explanatory.

E: Cannot use fix gcmc in a 2d simulation

Fix gcmc is set up to run in 3d only. No 2d simulations with fix gcmc
are allowed.

E: Could not find fix gcmc exclusion group ID

Self-explanatory.

E: Could not find fix gcmc rotation group ID

Self-explanatory.

E: Illegal fix gcmc gas mass <= 0

The computed mass of the designated gas molecule or atom type was less
than or equal to zero.

E: Cannot do GCMC on atoms in atom_modify first group

This is a restriction due to the way atoms are organized in a list to
enable the atom_modify first command.

E: Could not find specified fix gcmc group ID

Self-explanatory.

E: Fix gcmc put atom outside box

This should not normally happen.  Contact the developers.

E: Fix gcmc ran out of available molecule IDs

See the setting for tagint in the src/lmptype.h file.

E: Fix gcmc ran out of available atom IDs

See the setting for tagint in the src/lmptype.h file.

E: Too many total atoms

See the setting for bigint in the src/lmptype.h file.

*/
