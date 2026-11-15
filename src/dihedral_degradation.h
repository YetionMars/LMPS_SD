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

#ifndef LMP_DIHEDRAL_DEGRADATION_H
#define LMP_DIHEDRAL_DEGRADATION_H

namespace LAMMPS_NS {

// Internal interface returned by Fix::extract("dihedral_degradation", dim).
// Force styles query layer strength without depending on the MC fix class.
class DihedralDegradation {
 public:
  virtual ~DihedralDegradation() noexcept(false) = default;
  virtual double dihedral_scale(int atom_type) const = 0;
};

}    // namespace LAMMPS_NS

#endif
