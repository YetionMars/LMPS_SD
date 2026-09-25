/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(addforceXY,FixAddForceXY);  //1.1 Modify register name. FixStyle(addforce,FixAddForce)
// clang-format on
#else

#ifndef LMP_FIX_ADDFORCEXY_H  // 1.1 Modify protective macro name. LMP_FIX_ADDFORCE_H
#define LMP_FIX_ADDFORCEXY_H  // 1.1 

#include "fix.h"

namespace LAMMPS_NS {

class FixAddForceXY : public Fix {  // 1.1 Define derived class name. class FixAddForce
 public:
  FixAddForceXY(class LAMMPS *, int, char **);  // 1.1 Define derived class name
  ~FixAddForceXY() override;  // 1.1 Define derived class name
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  double compute_vector(int) override;
  double memory_usage() override;

  enum { NONE, CONSTANT, EQUAL, ATOM };  // This command in old script is set in cpp file.

 protected:
  double xvalue, yvalue, zvalue;
  int varflag;
  char *xstr, *ystr, *zstr, *estr;
  char *idregion;
  class Region *region;
  int xvar, yvar, zvar, evar, xstyle, ystyle, zstyle, estyle;
  double foriginal[4], foriginal_all[4];
  int force_flag;
  int ilevel_respa;

// 1.3 Define new variables for developed functions
  double x0,y0;			// 1.3 Variable: rotation center
  double kx,ky;			// 1.3 Variable: spring constants
  char *kxstr,*kystr;	// 1.3 Variable: store the variable names used for kx and ky
  int kxstyle,kystyle,kxvar,kyvar;	// 1.3 Variable: read force components as constant values or variables
  double *ework;		// 1.3 Variable: 1D array to store the work done by the added force

  int maxatom;
  double **sforce;  // Orinal cpp command, 2D array, be replaced by ework.
};

}    // namespace LAMMPS_NS

#endif
#endif
