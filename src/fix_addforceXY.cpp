/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_addforceXY.h"  // 2.1 Add header file

#include "atom.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAddForceXY::FixAddForceXY(LAMMPS *lmp, int narg, char **arg) :  // 2.2 Modify class name
    Fix(lmp, narg, arg), xstr(nullptr), ystr(nullptr), zstr(nullptr), estr(nullptr),
    idregion(nullptr), region(nullptr), sforce(nullptr), kxstr(nullptr), kystr(nullptr), 
	ework(nullptr) // 2.3 Define new variables
{
  if (narg < 8) utils::missing_cmd_args(FLERR, "fix addforce", error);  // 2.4 Change narg number

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  xstyle = ystyle = zstyle = NONE;
  xvar = yvar = zvar = -1;
  varflag = NONE; 
  kxstr = kystr = nullptr; // 2.5 Declare as NULL again.

/* These variables, x0, kx, y0, and ky, are parsed from the LAMMPS input script */
  if (utils::strmatch(arg[3], "^v_")) {
    xstr = utils::strdup(arg[3] + 2);
  } else {
    x0 = utils::numeric(FLERR, arg[3], false, lmp);  // 2.6 Change the target variable value name
    xstyle = CONSTANT;
  }
  if (utils::strmatch(arg[4], "^v_")) {
    ystr = utils::strdup(arg[4] + 2);
  } else {
    y0 = utils::numeric(FLERR, arg[4], false, lmp);  // 2.6 Change the target variable value name
    ystyle = CONSTANT;
  }
  if (utils::strmatch(arg[5], "^v_")) {
    zstr = utils::strdup(arg[5] + 2);
  } else {
    zvalue = utils::numeric(FLERR, arg[5], false, lmp);
    zstyle = CONSTANT;
  }

/* 2.7 Add new variables for the new command */
  if (utils::strmatch(arg[6], "^v_")) {
    kxstr = utils::strdup(arg[6] + 2);
  } else {
    kx = utils::numeric(FLERR, arg[6], false, lmp);
    kxstyle = CONSTANT;
  }
  if (utils::strmatch(arg[7], "^v_")) {
    kystr = utils::strdup(arg[7] + 2);
  } else {
    ky = utils::numeric(FLERR, arg[7], false, lmp);
    kystyle = CONSTANT;
  }
 /* 2.7 Add new variables for the new command */
 
  // optional args

  nevery = 1;
  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "every") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix addforce every", error);
      nevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (nevery <= 0) error->all(FLERR, "Invalid fix addforce every argument: {}", nevery);
      iarg += 2;
    } else if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix addforce region", error);
      region = domain->get_region_by_id(arg[iarg + 1]);
      if (!region) error->all(FLERR, "Region {} for fix addforce does not exist", arg[iarg + 1]);
      delete[] idregion;
      idregion = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "energy") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix addforce energy", error);
      if (utils::strmatch(arg[iarg + 1], "^v_")) {
        delete[] estr;
        estr = utils::strdup(arg[iarg + 1] + 2);
      } else
        error->all(FLERR, "Invalid fix addforceXY energy argument: {}", arg[iarg + 1]);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix addforceXY keyword: {}", arg[iarg]);
  }

  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

  maxatom = 1;
  memory->create(sforce, maxatom, 4, "addforceXY:sforce");
  memory->create(ework,1,"addforceXY::ework");  // 2.8 Add assignment of the memory to new target
}

/* ---------------------------------------------------------------------- */

FixAddForceXY::~FixAddForceXY()
{
  if (copymode) return;
  delete[] xstr;
  delete[] ystr;
  delete[] zstr;
  delete[] estr;
  delete[] idregion;
  memory->destroy(sforce);
  memory->destroy(ework);  // SM
}

/* ---------------------------------------------------------------------- */

int FixAddForceXY::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddForceXY::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0) error->all(FLERR, "Variable {} for fix addforceXY does not exist", xstr);
    if (input->variable->equalstyle(xvar))
      xstyle = EQUAL;
    //else if (input->variable->atomstyle(xvar))
      //xstyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addforceXY is invalid style", xstr);
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0) error->all(FLERR, "Variable {} for fix addforceXY does not exist", ystr);
    if (input->variable->equalstyle(yvar))
      ystyle = EQUAL;
    //else if (input->variable->atomstyle(yvar))
      //ystyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addforceXY is invalid style", ystr);
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0) error->all(FLERR, "Variable {} for fix addforceXY does not exist", zstr);
    if (input->variable->equalstyle(zvar))
      zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar))
      zstyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addforceXY is invalid style", zstr);
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0) error->all(FLERR, "Variable {} for fix addforceXY does not exist", estr);
    if (input->variable->atomstyle(evar))
      estyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addforceXY is invalid style", estr);
  } else
    estyle = NONE;

/* 3.1 Initialize kxstr and kystr */
  if (kxstr) {
    kxvar = input->variable->find(kxstr);
    if (kxvar < 0) error->all(FLERR, "Variable {} for fix addforceXY does not exist", kxstr);
    if (input->variable->equalstyle(kxvar))
      kxstyle = EQUAL;
    else
      error->all(FLERR, "Variable {} for fix addforceXY is invalid style", kxstr);
  }  // else kxstyle = NONE;

  if (kystr) {
    kyvar = input->variable->find(kystr);
    if (kyvar < 0) error->all(FLERR, "Variable {} for fix addforceXY does not exist", kystr);
    if (input->variable->equalstyle(kyvar))
      kystyle = EQUAL;
    else
      error->all(FLERR, "Variable {} for fix addforceXY is invalid style", kystr);
  }  // else kystyle = NONE;
/* 3.1 Initialize kxstr and kystr */

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region {} for fix addforceXY does not exist", idregion);
  }

/* 3.2 Flag updating */
  if (zstyle == ATOM)  // x0, y0, kx, ky are not entereed as atom style variables
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL || kxstyle == EQUAL || kystyle == EQUAL)
    varflag = EQUAL;
  else
    varflag = CONSTANT;

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR, "Cannot use variable energy with constant force in fix addforceXY");
  if ((varflag == EQUAL || varflag == ATOM) && update->whichflag == 2 && estyle == NONE)
    error->all(FLERR, "Must use variable energy with fix addforceXY");

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixAddForceXY::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixAddForceXY::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddForceXY::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  double v[6];
  int nlocal = atom->nlocal;

  if (update->ntimestep % nevery) return;

  // virial setup

  v_init(vflag);

  // update region if necessary

  if (region) region->prematch();

  // reallocate sforce array if necessary

  if ((varflag == ATOM || estyle == ATOM) && atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(sforce);
    memory->destroy(ework);  // 4.1 Add memory destroy for ework
    memory->create(sforce, maxatom, 4, "addforceXY:sforce");
    memory->create(ework,1,"addforceXY::ework");  // 4.1 Add memory storage for ework, 1D array of length 1. Not very general, but can be used.
  }

  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added

  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  // constant force
  // potential energy = - x dot f in unwrapped coords

  if (varflag == CONSTANT) {
    double unwrap[3];
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0], x[i][1], x[i][2])) continue;
        domain->unmap(x[i], image[i], unwrap);
        foriginal[0] -= -kx*(x[i][0] - x0)*unwrap[0] - ky*(x[i][1] - y0)*unwrap[1];  //  4.2 Energy contribution from the fix force
        foriginal[1] += f[i][0];
        foriginal[2] += f[i][1];
        foriginal[3] += f[i][2];

    /* 4.3 Calculating forces to apply to the designed atoms */
    /*
        f[i][0] += xvalue;
        f[i][1] += yvalue;
        f[i][2] += zvalue;
    */
        f[i][0] += -kx*(x[i][0] - x0);
        f[i][1] += -ky*(x[i][1] - y0);
        f[i][2] += 0.0;

    /* 4.4 Calculating virial contributions from the fix force */
        if (evflag) {
          v[0] = -kx*(x[i][0] - x0) * unwrap[0];
          v[1] = -ky*(x[i][1] - y0) * unwrap[1];
          v[2] = 0.0 * unwrap[2];
          v[3] = -kx*(x[i][0] - x0) * unwrap[1];
          v[4] = -kx*(x[i][0] - x0) * unwrap[2];
          v[5] = -ky*(x[i][1] - y0) * unwrap[2];
          v_tally(i, v);
        }
      }

    // variable force, wrap with clear/add
    // potential energy = evar if defined, else 0.0
    // wrap with clear/add

  } else {
    double unwrap[3];

    modify->clearstep_compute();

    /* 4.5 Modify variables name */
    if (xstyle == EQUAL)
      x0 = input->variable->compute_equal(xvar);
    //else if (xstyle == ATOM)
    //  input->variable->compute_atom(xvar, igroup, &sforce[0][0], 4, 0);
    if (ystyle == EQUAL)
      y0 = input->variable->compute_equal(yvar);
    //else if (ystyle == ATOM)
    //  input->variable->compute_atom(yvar, igroup, &sforce[0][1], 4, 0);
    if (zstyle == EQUAL)
      zvalue = input->variable->compute_equal(zvar);
    //else if (zstyle == ATOM)
    //  input->variable->compute_atom(zvar, igroup, &sforce[0][2], 4, 0);
    if (estyle == ATOM) //input->variable->compute_atom(evar, igroup, &sforce[0][3], 4, 0);
      input->variable->compute_atom(evar,igroup,&ework[0],1,0);

    if (kxstyle == EQUAL) kx = input->variable->compute_equal(kxvar);
    if (kystyle == EQUAL) ky = input->variable->compute_equal(kyvar);
    /* 4.5 Modify variables name */

    modify->addstep_compute(update->ntimestep + 1);

    /* 4.6 Modify energy and force components variable names */
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0], x[i][1], x[i][2])) continue;
        domain->unmap(x[i], image[i], unwrap);
        //if (xstyle == ATOM) xvalue = sforce[i][0];
        //if (ystyle == ATOM) yvalue = sforce[i][1];
        //if (zstyle == ATOM) zvalue = sforce[i][2];

        if (estyle == ATOM) {
          foriginal[0] += ework[0];
        } else {
          if (xstyle) foriginal[0] -= -kx*(x[i][0] - x0)*unwrap[0];
          if (ystyle) foriginal[0] -= -ky*(x[i][1] - y0)*unwrap[1];
          if (zstyle) foriginal[0] -= 0.0;
        }
        foriginal[1] += f[i][0];
        foriginal[2] += f[i][1];
        foriginal[3] += f[i][2];

        if (xstyle) f[i][0] += -kx*(x[i][0] - x0);
        if (ystyle) f[i][1] += -ky*(x[i][1] - y0);
        if (zstyle) f[i][2] += 0.0;
        if (evflag) {
          v[0] = xstyle ? (-kx*(x[i][0] - x0))*unwrap[0] : 0.0;
          v[1] = ystyle ? (-ky*(x[i][1] - y0))*unwrap[1] : 0.0;
          v[2] = zstyle ? 0.0*unwrap[2] : 0.0;
          v[3] = xstyle ? (-kx*(x[i][0] - x0))*unwrap[1] : 0.0;
          v[4] = xstyle ? (-kx*(x[i][0] - x0))*unwrap[2] : 0.0;
          v[5] = ystyle ? (-ky*(x[i][1] - y0))*unwrap[2] : 0.0;
          v_tally(i, v);
          /* 4.6 Modify energy and force components variable names */
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixAddForceXY::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddForceXY::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixAddForceXY::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal, foriginal_all, 4, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return foriginal_all[0];
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixAddForceXY::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal, foriginal_all, 4, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return foriginal_all[n + 1];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAddForceXY::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = maxatom * 4 * sizeof(double);
  return bytes;
}
