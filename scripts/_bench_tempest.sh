#!/bin/bash
# Time TempestRemap conservative FV->FV weight generation (mesh gen + overlap + offline map),
# the same path that produced map_*_conserve.nc. CPU, single process.
cd ~/remapgnn
BIN=$(pwd)/software/tempestremap-install/bin
S=/gpfs/fs1/soft/swing/spack-0.16.1/opt/spack/linux-ubuntu20.04-x86_64
MKLDIR=$(find $S -path '*intel64/libmkl_rt.so.2' 2>/dev/null | head -1 | xargs dirname)
GCCDIR=$(find $S -path '*gcc-8.5.0*/lib64/libstdc++.so.6' 2>/dev/null | head -1 | xargs dirname)
export LD_LIBRARY_PATH=$S/gcc-8.5.0/netcdf-c-4.7.4-y6twdyt/lib:$S/gcc-8.5.0/hdf5-1.10.7-zjwj3y2/lib:$S/gcc-8.5.0/openmpi-4.1.1-by6rv67/lib:$MKLDIR:$GCCDIR
TMP=$(mktemp -d)

el() { python3 -c "import sys;print('%.3f'%(float(sys.argv[2])-float(sys.argv[1])))" "$1" "$2"; }
now() { date +%s.%N; }

bench() {
  R=$1; LAB=$2
  cs=$TMP/cs$R.g; ic=$TMP/ic$R.g; ov=$TMP/ov_$LAB.g; mp=$TMP/map_$LAB.nc
  echo ""
  echo "=== $LAB  (CS res=$R -> ICOD res=$R) ==="
  $BIN/GenerateCSMesh  --res $R --file $cs        >$TMP/log 2>&1
  $BIN/GenerateICOMesh --res $R --dual --file $ic >>$TMP/log 2>&1
  t0=$(now); $BIN/GenerateOverlapMesh --a $cs --b $ic --out $ov >>$TMP/log 2>&1; ec1=$?; t1=$(now)
  t2=$(now); $BIN/GenerateOfflineMap --in_mesh $cs --out_mesh $ic --ov_mesh $ov \
              --in_type fv --out_type fv --in_np 1 --out_map $mp >>$TMP/log 2>&1; ec2=$?; t3=$(now)
  ov_t=$(el $t0 $t1); mp_t=$(el $t2 $t3); tot=$(el $t0 $t3)
  echo "  overlap mesh:    $ov_t s   (exit $ec1)"
  echo "  offline map:     $mp_t s   (exit $ec2)"
  echo "  WEIGHT-GEN TOTAL: $tot s"
  ls -la $mp 2>/dev/null | awk '{print "  map file:", $5, "bytes"}'
}

bench 32 r32
bench 64 r64
bench 128 r128
bench 256 r256
echo ""
echo "TEMPEST_BENCH_DONE"
tail -2 $TMP/log
rm -rf "$TMP"
