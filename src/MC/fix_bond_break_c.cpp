// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.


   bond/break/c V1.3
   Syntax: fix ID group-ID bond/break/c Nevery btype Rmax Rmin [prob fraction seed]
   Every Nevery steps, consider btype bonds with both atoms in the group.
   r > Rmax permanently deletes the bond and affected higher-order topology.
   r < Rmin marks the common atom type of the two endpoints as damaged.
   All harmonic dihedrals whose four atom types match that damaged type use
   DEGRADATION_RATIO times the original K, including atoms outside the group.
   Bonds, original coefficients, topology, and special neighbors are retained
   after compression. Mixed-type bonds do not trigger compression damage.
   Damaged layers skip later compression events but remain eligible for
   tensile deletion. Equality to either threshold does not trigger an event.

   The compile-time ratio is in [0,1]. Zero disables affected dihedral terms;
   one records damage without changing their strength. Damage never compounds.
   The internal DihedralDegradation interface separates this MC fix from the
   harmonic force style. Only plain dihedral_style harmonic consumes it.

   Each rank stores the complete damaged atom-type set; sorting and migration
   do not change layer identity. Restart stores that set, event counters, and
   per-rank random states. Recreate the same fix ID/style/btype after restart.
   The current compile-time ratio is applied to restored damage. Ordinary
   write_data omits damage; unfix removes its weakening effect. Restart format
   3 stores separate tensile and compression counts; earlier formats do not
   contain the required separate history. Random trajectories
   are only reproducible with unchanged rank count and atom ordering.

   f_ID[1:2] report recent/cumulative tensile deletions; f_ID[3:4] report
   recent/cumulative compression triggers. f_ID[5] is the cumulative sum and
   f_ID[6] is the current global bond count. Recent counts retain the last
   check's values between checks. Multiple compression bonds in one check may
   mark the same layer, so trigger counts are not layer or dihedral counts.

------------------------------------------------------------------------- */

#include "fix_bond_break_c.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "graphics.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

#include "fix_bond_history.h"

#include <algorithm>
#include <cstring>
#include <limits>

using namespace LAMMPS_NS;
using namespace FixConst;

// Developer interface: retained dihedral strength in a damaged layer.
// Change this constant and rebuild to compare degradation ratios.
static constexpr double DEGRADATION_RATIO = 0.03;
static_assert(DEGRADATION_RATIO >= 0.0 && DEGRADATION_RATIO <= 1.0,
              "DEGRADATION_RATIO must be between 0 and 1");

static constexpr int DELTA = 16;

/* ---------------------------------------------------------------------- */

FixBondBreakC::FixBondBreakC(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  partner(nullptr), finalpartner(nullptr), score(nullptr), probability(nullptr),
  broken(nullptr), copy(nullptr), random(nullptr), imgobjs(nullptr), imgparms(nullptr)
{
  if (narg < 7) utils::missing_cmd_args(FLERR, "fix bond/break/c", error);

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, 3, "Fix bond/break/c Nevery must be positive");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 6;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;

  vizsteps = 1000;

  btype = utils::expand_type_int(FLERR, arg[4], Atom::BOND, lmp);
  maxcutoff = utils::numeric(FLERR, arg[5], false, lmp);
  mincutoff = utils::numeric(FLERR, arg[6], false, lmp);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/break/c command");
  if (maxcutoff <= 0.0)
    error->all(FLERR, 5, "Fix bond/break/c Rmax must be positive");
  if ((mincutoff < 0.0) || (mincutoff >= maxcutoff))
    error->all(FLERR, 6, "Fix bond/break/c Rmin must satisfy 0 <= Rmin < Rmax");

  maxcutsq = maxcutoff*maxcutoff;
  mincutsq = mincutoff*mincutoff;

  // optional keywords

  fraction = 1.0;
  int seed = 12345;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+3 > narg) utils::missing_cmd_args(FLERR, "fix bond/break/c prob", error);
      fraction = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      seed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if ((fraction < 0.0) || (fraction > 1.0))
        error->all(FLERR, iarg+1, "Fix bond/break/c probability must be between 0 and 1");
      if (seed <= 0)
        error->all(FLERR, iarg+2, "Fix bond/break/c random seed must be positive");
      iarg += 3;
    } else error->all(FLERR, iarg, "Unknown fix bond/break/c keyword {}", arg[iarg]);
  }

  // error check

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/break/c with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

  comm_forward = MAX(2,2+atom->maxspecial);
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;

  maxbreak = 0;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // zero out stats

  tensile_count = tensile_count_total = 0;
  compression_count = compression_count_total = 0;
}

/* ---------------------------------------------------------------------- */

FixBondBreakC::~FixBondBreakC()
{
  delete random;

  // delete locally stored arrays

  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(score);
  memory->destroy(broken);
  delete[] copy;
  memory->destroy(imgobjs);
  memory->destroy(imgparms);
}

/* ---------------------------------------------------------------------- */

int FixBondBreakC::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"vizsteps") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify bond/break/c", error);
    vizsteps = utils::inumeric(FLERR, arg[1], false, lmp);
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixBondBreakC::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakC::init()
{
  // Other styles (including accelerated harmonic variants) do not consume
  // the internal damage record. Tensile-only use remains unrestricted.

  if ((mincutoff > 0.0) || !degraded_types.empty()) {
    if (!force->dihedral || strcmp(force->dihedral_style,"harmonic") != 0)
      error->all(FLERR,"Fix bond/break/c compression degradation requires dihedral_style harmonic");
  }

  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;

  // enable angle/dihedral/improper breaking if any defined

  if (atom->nangles) angleflag = 1;
  else angleflag = 0;
  if (atom->ndihedrals) dihedralflag = 1;
  else dihedralflag = 0;
  if (atom->nimpropers) improperflag = 1;
  else improperflag = 0;

  if (force->improper) {
    if (force->improper_match("^class2") || force->improper_match("^ring"))
      error->all(FLERR,"Cannot yet use fix bond/break/c with this improper style");
  }

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakC::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,type;
  int ndelete,deletecount,eventcount;
  double delx,dely,delz,rsq,candidate_score;
  tagint candidate_partner;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  if (lastcheck < neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // resize bond partner list and initialize it
  // probability array overlays score array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(score);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/break/c:partner");
    memory->create(finalpartner,nmax,"bond/break/c:finalpartner");
    memory->create(score,nmax,"bond/break/c:score");
    probability = score;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    score[i] = 0.0;
  }

  // loop over bond list
  // setup possible partner list of bonds to break

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *atom_type = atom->type;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  auto breakfixes = modify->get_fix_by_style("^bond/break/c$");

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type != btype) continue;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    domain->minimum_image(FLERR,delx,dely,delz);
    rsq = delx*delx + dely*dely + delz*delz;

    // Use a dimensionless squared exceedance score so tensile and
    // compressive candidates can be compared on the same scale.

    if (rsq > maxcutsq) {
      candidate_score = rsq / maxcutsq;
    } else if (rsq < mincutsq) {
      // The atom type identifies the whole layer. Only compression is
      // suppressed after damage; tensile deletion above remains available.
      if (atom_type[i1] != atom_type[i2]) continue;
      bool degraded = false;
      for (auto *ifix : breakfixes)
        if (static_cast<FixBondBreakC *>(ifix)->degraded_types.count(atom_type[i1])) {
          degraded = true;
          break;
        }
      if (degraded) continue;
      if (rsq == 0.0)
        candidate_score = std::numeric_limits<double>::max();
      else
        candidate_score = mincutsq / rsq;
    } else continue;

    // For equal scores, choose the smaller atom ID so the result does not
    // depend on the order in which bonds appear in the local bond list.

    candidate_partner = tag[i2];
    if (candidate_score > score[i1] ||
        (candidate_score == score[i1] &&
         (partner[i1] == 0 || candidate_partner < partner[i1]))) {
      partner[i1] = candidate_partner;
      score[i1] = candidate_score;
    }

    candidate_partner = tag[i1];
    if (candidate_score > score[i2] ||
        (candidate_score == score[i2] &&
         (partner[i2] == 0 || candidate_partner < partner[i2]))) {
      partner[i2] = candidate_partner;
      score[i2] = candidate_score;
    }
  }

  // reverse comm of partner info

  if (force->newton_bond) comm->reverse_comm(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  if (fraction < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }

  commflag = 1;
  comm->forward_comm(this,2);

  // find instances of bond history to delete data
  auto histories = modify->get_fix_by_style("BOND_HISTORY");
  int n_histories = histories.size();

  // break bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied

  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  nbreak = 0;
  ndelete = 0;
  std::set<int> newly_degraded;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // apply probability constraint using RN for atom with smallest ID

    if (fraction < 1.0) {
      if (tag[i] < tag[j]) {
        if (probability[i] >= fraction) continue;
      } else {
        if (probability[j] >= fraction) continue;
      }
    }

    // Recompute the selected bond length before changing its topology.
    // Candidate selection and bond modification occur in the same call, so
    // no integration step has changed the coordinates in between.

    delx = x[i][0] - x[j][0];
    dely = x[i][1] - x[j][1];
    delz = x[i][2] - x[j][2];
    domain->minimum_image(FLERR,delx,dely,delz);
    rsq = delx*delx + dely*dely + delz*delz;

    const bool compression_failure = (rsq < mincutsq);

    if (compression_failure) {

      // Record the layer on the owner of the smaller atom ID. Keep the
      // bond and all topology; harmonic dihedrals query the layer strength.
      // Publish only after this batch so event counts do not depend on order.

      if (tag[i] < tag[j]) newly_degraded.insert(atom_type[i]);

    } else {

      // Tensile failure: retain the original fix bond/break behavior and
      // permanently remove the selected bond from atom I if I stores it.

      for (m = 0; m < num_bond[i]; m++) {
        if (bond_atom[i][m] == partner[i]) {
          for (k = m; k < num_bond[i]-1; k++) {
            bond_atom[i][k] = bond_atom[i][k+1];
            bond_type[i][k] = bond_type[i][k+1];
            if (n_histories > 0)
              for (auto &ihistory: histories)
                dynamic_cast<FixBondHistory *>(ihistory)->shift_history(i,k,k+1);
          }
          if (n_histories > 0)
            for (auto &ihistory: histories)
              dynamic_cast<FixBondHistory *>(ihistory)->delete_history(i,num_bond[i]-1);
          num_bond[i]--;
          break;
        }
      }

      // Remove J from the special list only for a permanently deleted bond.

      slist = special[i];
      n1 = nspecial[i][0];
      for (m = 0; m < n1; m++)
        if (slist[m] == partner[i]) break;
      n3 = nspecial[i][2];
      for (; m < n3-1; m++) slist[m] = slist[m+1];
      nspecial[i][0]--;
      nspecial[i][1]--;
      nspecial[i][2]--;

      // Only tensile deletion is forwarded to update_topology(), where
      // affected angles, dihedrals, and impropers are removed.

      finalpartner[i] = tag[j];
      finalpartner[j] = tag[i];
      if (tag[i] < tag[j]) ndelete++;
    }

    // Count both tensile deletion and compression degradation as one event.

    if (tag[i] < tag[j]) nbreak++;

    // record atoms involved in broken bond
    if (vizsteps > 0) {
      vizatoms[tag[i]] = vizsteps;
      vizatoms[tag[j]] = vizsteps;
    }
  }

  // tally stats

  MPI_Allreduce(&nbreak,&eventcount,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&ndelete,&deletecount,1,MPI_INT,MPI_SUM,world);
  tensile_count = deletecount;
  compression_count = eventcount-deletecount;
  tensile_count_total += tensile_count;
  compression_count_total += compression_count;
  atom->nbonds -= deletecount;

  // Replicate new records before atom migration and force evaluation. All
  // ranks then recognize damage regardless of which rank owns either atom.

  if (compression_count) synchronize_degraded_types(newly_degraded);

  // if visualization support is enabled, age vizatoms and remove expired ones
  if (vizsteps > 0) {
    std::vector<tagint> eraseme;
    for (const auto &[key, value] : vizatoms) {
      int idx = atom->map(key);
      if ((idx < 0) || (value < 0)) {
        eraseme.push_back(key);
        continue;
      }
      vizatoms[key] = value - nevery;
    }
    for (const auto &key : eraseme) vizatoms.erase(key);
  }

  // trigger reneighboring if any bonds were broken
  // this ensures neigh lists will immediately reflect the topology changes
  // done if no bonds broken

  if (deletecount) next_reneighbor = update->ntimestep;
  if (!deletecount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds

  commflag = 2;
  comm->forward_comm(this);

  // Hide deleted bonds from later fix instances before the immediate
  // neighbor rebuild. Topology removal below uses finalpartner and broken.
  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    if ((bondlist[n][2] == btype) && (finalpartner[i1] == tag[i2]) &&
        (finalpartner[i2] == tag[i1]))
      bondlist[n][2] = 0;
  }

  // create list of broken bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to ensure a broken bond to unknown atom is included
  //   i.e. bond partner outside of cutoff length

  nbreak = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (nbreak == maxbreak) {
        maxbreak += DELTA;
        memory->grow(broken,maxbreak,2,"bond/break/c:broken");
      }
      broken[nbreak][0] = tag[i];
      broken[nbreak][1] = finalpartner[i];
      nbreak++;
    }
  }

  // update special neigh lists of all atoms affected by any broken bond
  // also remove angles/dihedrals/impropers broken by broken bonds

  update_topology();


}

/* ----------------------------------------------------------------------
   ensure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly deleted
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is deleted
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixBondBreakC::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR,"Fix bond/break/c needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and broken bonds
   influenced = 1 if atom's topology is affected by any broken bond
     yes if is one of 2 atoms in bond
     yes if both atom IDs appear in atom's special list
     else no
   if influenced:
     check for angles/dihedrals/impropers to break due to specific broken bonds
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixBondBreakC::update_topology()
{
  int i,j,k,n,influence,influenced,found;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < nbreak; j++) {
      id1 = broken[j][0];
      id2 = broken[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][2];
        found = 0;
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) found++;
        if (found == 2) influence = 1;
      }
      if (!influence) continue;
      influenced = 1;

      if (angleflag) break_angles(i,id1,id2);
      if (dihedralflag) break_dihedrals(i,id1,id2);
      if (improperflag) break_impropers(i,id1,id2);
    }

    if (influenced) rebuild_special_one(i);
  }

  int newton_bond = force->newton_bond;

  int all;
  if (angleflag) {
    MPI_Allreduce(&nangles,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 3;
    atom->nangles -= all;
  }
  if (dihedralflag) {
    MPI_Allreduce(&ndihedrals,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->ndihedrals -= all;
  }
  if (improperflag) {
    MPI_Allreduce(&nimpropers,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->nimpropers -= all;
  }
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

void FixBondBreakC::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(tagint));
}

/* ----------------------------------------------------------------------
   break any angles owned by atom M that include atom IDs 1 and 2
   angle is broken if ID1-ID2 is one of 2 bonds in angle (I-J,J-K)
------------------------------------------------------------------------- */

void FixBondBreakC::break_angles(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_angle = atom->num_angle[m];
  int *angle_type = atom->angle_type[m];
  tagint *angle_atom1 = atom->angle_atom1[m];
  tagint *angle_atom2 = atom->angle_atom2[m];
  tagint *angle_atom3 = atom->angle_atom3[m];

  int i = 0;
  while (i < num_angle) {
    found = 0;
    if (angle_atom1[i] == id1 && angle_atom2[i] == id2) found = 1;
    else if (angle_atom2[i] == id1 && angle_atom3[i] == id2) found = 1;
    else if (angle_atom1[i] == id2 && angle_atom2[i] == id1) found = 1;
    else if (angle_atom2[i] == id2 && angle_atom3[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_angle-1; j++) {
        angle_type[j] = angle_type[j+1];
        angle_atom1[j] = angle_atom1[j+1];
        angle_atom2[j] = angle_atom2[j+1];
        angle_atom3[j] = angle_atom3[j+1];
      }
      num_angle--;
      nangles++;
    }
  }

  atom->num_angle[m] = num_angle;
}

/* ----------------------------------------------------------------------
   break any dihedrals owned by atom M that include atom IDs 1 and 2
   dihedral is broken if ID1-ID2 is one of 3 bonds in dihedral (I-J,J-K.K-L)
------------------------------------------------------------------------- */

void FixBondBreakC::break_dihedrals(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_dihedral = atom->num_dihedral[m];
  int *dihedral_type = atom->dihedral_type[m];
  tagint *dihedral_atom1 = atom->dihedral_atom1[m];
  tagint *dihedral_atom2 = atom->dihedral_atom2[m];
  tagint *dihedral_atom3 = atom->dihedral_atom3[m];
  tagint *dihedral_atom4 = atom->dihedral_atom4[m];

  int i = 0;
  while (i < num_dihedral) {
    found = 0;
    if (dihedral_atom1[i] == id1 && dihedral_atom2[i] == id2) found = 1;
    else if (dihedral_atom2[i] == id1 && dihedral_atom3[i] == id2) found = 1;
    else if (dihedral_atom3[i] == id1 && dihedral_atom4[i] == id2) found = 1;
    else if (dihedral_atom1[i] == id2 && dihedral_atom2[i] == id1) found = 1;
    else if (dihedral_atom2[i] == id2 && dihedral_atom3[i] == id1) found = 1;
    else if (dihedral_atom3[i] == id2 && dihedral_atom4[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_dihedral-1; j++) {
        dihedral_type[j] = dihedral_type[j+1];
        dihedral_atom1[j] = dihedral_atom1[j+1];
        dihedral_atom2[j] = dihedral_atom2[j+1];
        dihedral_atom3[j] = dihedral_atom3[j+1];
        dihedral_atom4[j] = dihedral_atom4[j+1];
      }
      num_dihedral--;
      ndihedrals++;
    }
  }

  atom->num_dihedral[m] = num_dihedral;
}

/* ----------------------------------------------------------------------
   break any impropers owned by atom M that include atom IDs 1 and 2
   improper is broken if ID1-ID2 is one of 3 bonds in improper (I-J,I-K,I-L)
------------------------------------------------------------------------- */

void FixBondBreakC::break_impropers(int m, tagint id1, tagint id2)
{
  int j,found;

  int num_improper = atom->num_improper[m];
  int *improper_type = atom->improper_type[m];
  tagint *improper_atom1 = atom->improper_atom1[m];
  tagint *improper_atom2 = atom->improper_atom2[m];
  tagint *improper_atom3 = atom->improper_atom3[m];
  tagint *improper_atom4 = atom->improper_atom4[m];

  int i = 0;
  while (i < num_improper) {
    found = 0;
    if (improper_atom1[i] == id1 && improper_atom2[i] == id2) found = 1;
    else if (improper_atom1[i] == id1 && improper_atom3[i] == id2) found = 1;
    else if (improper_atom1[i] == id1 && improper_atom4[i] == id2) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom2[i] == id1) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom3[i] == id1) found = 1;
    else if (improper_atom1[i] == id2 && improper_atom4[i] == id1) found = 1;
    if (!found) i++;
    else {
      for (j = i; j < num_improper-1; j++) {
        improper_type[j] = improper_type[j+1];
        improper_atom1[j] = improper_atom1[j+1];
        improper_atom2[j] = improper_atom2[j+1];
        improper_atom3[j] = improper_atom3[j+1];
        improper_atom4[j] = improper_atom4[j+1];
      }
      num_improper--;
      nimpropers++;
    }
  }

  atom->num_improper[m] = num_improper;
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int FixBondBreakC::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakC::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondBreakC::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  if (commflag == 1) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
      buf[m++] = probability[j];
    }
    return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakC::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  if (commflag == 1) {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      partner[i] = (tagint) ubuf(buf[m++]).i;
      probability[i] = buf[m++];
    }

  } else {

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondBreakC::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = score[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakC::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  tagint received_partner;
  double received_score;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    received_partner = (tagint) ubuf(buf[m]).i;
    received_score = buf[m+1];

    // Merge ghost candidates into the owner using the same score and
    // atom-ID tie-breaking rule as the local candidate selection.

    if (received_partner != 0 &&
        (received_score > score[j] ||
         (received_score == score[j] &&
          (partner[j] == 0 || received_partner < partner[j])))) {
      partner[j] = received_partner;
      score[j] = received_score;
    }
    m += 2;
  }
}

/* ---------------------------------------------------------------------- */

double FixBondBreakC::compute_vector(int n)
{
  if (n == 0) return static_cast<double>(tensile_count);
  if (n == 1) return static_cast<double>(tensile_count_total);
  if (n == 2) return static_cast<double>(compression_count);
  if (n == 3) return static_cast<double>(compression_count_total);
  if (n == 4) return static_cast<double>(tensile_count_total)+compression_count_total;
  return static_cast<double>(atom->nbonds);
}

/* ---------------------------------------------------------------------- */

void *FixBondBreakC::extract(const char *str, int &dim)
{
  if (strcmp(str,"dihedral_degradation") == 0) {
    dim = 0;
    return static_cast<DihedralDegradation *>(this);
  }
  return nullptr;
}

/* ---------------------------------------------------------------------- */

double FixBondBreakC::dihedral_scale(int atom_type) const
{
  return degraded_types.count(atom_type) ? DEGRADATION_RATIO : 1.0;
}

/* ----------------------------------------------------------------------
   publish newly damaged atom types on every rank before force evaluation
------------------------------------------------------------------------- */

void FixBondBreakC::synchronize_degraded_types(const std::set<int> &local)
{
  std::vector<int> flags(atom->ntypes+1,0);
  for (int type : local) flags[type] = 1;
  MPI_Allreduce(MPI_IN_PLACE,flags.data(),atom->ntypes+1,MPI_INT,MPI_MAX,world);
  for (int type = 1; type <= atom->ntypes; type++)
    if (flags[type]) degraded_types.insert(type);
}

/* ----------------------------------------------------------------------
   preserve layer state, counters, and each rank's probability stream
------------------------------------------------------------------------- */

void FixBondBreakC::write_restart(FILE *fp)
{
  double random_state[RanMars::STATE_SIZE];
  random->get_state(random_state);
  std::vector<double> states(me == 0 ? nprocs*RanMars::STATE_SIZE : 1);
  MPI_Gather(random_state,RanMars::STATE_SIZE,MPI_DOUBLE,states.data(),
             RanMars::STATE_SIZE,MPI_DOUBLE,0,world);
  if (me != 0) return;

  std::vector<double> values(8+degraded_types.size());
  values[0] = 3.0;    // separate tensile and compression event histories
  values[1] = btype;
  values[2] = tensile_count;
  values[3] = tensile_count_total;
  values[4] = compression_count;
  values[5] = compression_count_total;
  values[6] = static_cast<double>(degraded_types.size());
  values[7] = nprocs;
  size_t n = 8;
  for (int type : degraded_types) values[n++] = type;
  values.insert(values.end(),states.begin(),states.end());

  int size = static_cast<int>(values.size()*sizeof(double));
  fwrite(&size,sizeof(int),1,fp);
  fwrite(values.data(),sizeof(double),values.size(),fp);
}

/* ---------------------------------------------------------------------- */

void FixBondBreakC::restart(char *buf)
{
  auto *values = reinterpret_cast<double *>(buf);
  if ((values[0] != 3.0) || (values[1] != btype))
    error->all(FLERR,"Incompatible fix bond/break/c restart state or bond type (requires format 3)");
  tensile_count = static_cast<int>(values[2]);
  tensile_count_total = static_cast<int>(values[3]);
  compression_count = static_cast<int>(values[4]);
  compression_count_total = static_cast<int>(values[5]);
  const int ntypes = static_cast<int>(values[6]);
  const int saved_nprocs = static_cast<int>(values[7]);
  degraded_types.clear();
  for (int n = 0; n < ntypes; n++)
    degraded_types.insert(static_cast<int>(values[8+n]));

  // Existing ranks resume their own streams; extra ranks retain seeded streams.
  // A different rank count does not reproduce the original random trajectory.
  if (me < saved_nprocs) {
    double *states = values+8+ntypes;
    const int state_size = RanMars::state_size(states);
    random->set_state(states+me*state_size);
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondBreakC::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += (double)nmax * sizeof(double);
  bytes += degraded_types.size() * (sizeof(int) + 4*sizeof(void *));
  return bytes;
}

/* ----------------------------------------------------------------------
   provide graphics information to dump image to render spheres
   at the location of atoms that were involved in a reaction
------------------------------------------------------------------------- */

int FixBondBreakC::image(int *&objs, double **&parms)
{
  memory->destroy(imgobjs);
  memory->destroy(imgparms);

  int numobjs = vizatoms.size();
  int n = 0;
  if (numobjs > 0) {
    memory->create(imgobjs, numobjs, "bond/break/c:imgobjs");
    memory->create(imgparms, numobjs, 5, "bond/break/c:imgparms");

    int idx;
    const auto *const type = atom->type;
    const auto *const *const x = atom->x;
    for (const auto &[key, value] : vizatoms) {
      idx = atom->map(key);
      if (idx < 0) continue;
      imgobjs[n] = Graphics::SPHERE;
      imgparms[n][0] = type[idx];
      imgparms[n][1] = x[idx][0];
      imgparms[n][2] = x[idx][1];
      imgparms[n][3] = x[idx][2];
      imgparms[n][4] = 0.0;     // radius is set with fflag2 in dump image
      ++n;
    }
  }
  objs = imgobjs;
  parms = imgparms;
  return n;
}
