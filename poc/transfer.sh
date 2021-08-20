#!/bin/sh

l="
0,0,/home/lisergey/stride2/1.2.HC_X-5000_Y-10000_640_nm_4x_Left_000044.raw
0,1,/home/lisergey/stride2/1.2.HC_X-5000_Y-7500_640_nm_4x_Left_000041.raw
0,2,/home/lisergey/stride2/1.2.HC_X-5000_Y-5000_640_nm_4x_Left_000038.raw
0,3,/home/lisergey/stride2/1.2.HC_X-5000_Y-2500_640_nm_4x_Left_000035.raw
0,4,/home/lisergey/stride2/1.2.HC_X-5000_Y0_640_nm_4x_Left_000032.raw
1,0,/home/lisergey/stride2/1.2.HC_X-2500_Y-10000_640_nm_4x_Left_000029.raw
1,1,/home/lisergey/stride2/1.2.HC_X-2500_Y-7500_640_nm_4x_Left_000026.raw
1,2,/home/lisergey/stride2/1.2.HC_X-2500_Y-5000_640_nm_4x_Left_000023.raw
1,3,/home/lisergey/stride2/1.2.HC_X-2500_Y-2500_640_nm_4x_Left_000020.raw
1,4,/home/lisergey/stride2/1.2.HC_X-2500_Y0_640_nm_4x_Left_000017.raw
2,0,/home/lisergey/stride2/1.2.HC_X0_Y-10000_640_nm_4x_Right_000014.raw
2,1,/home/lisergey/stride2/1.2.HC_X0_Y-7500_640_nm_4x_Right_000011.raw
2,2,/home/lisergey/stride2/1.2.HC_X0_Y-5000_640_nm_4x_Right_000008.raw
2,3,/home/lisergey/stride2/1.2.HC_X0_Y-2500_640_nm_4x_Right_000005.raw
2,4,/home/lisergey/stride2/1.2.HC_X0_Y0_640_nm_4x_Right_000002.raw
"
SCRATCH="`ssh euler '. /etc/profile && echo "$SCRATCH"'`" || exit 1
for i in $l
do set -- `echo "$i" | sed 's/,/ /g'`
   printf 'rsync --compress-level=6 '\''%s'\'' euler:'\''%s'\''/stride2/%02dx%02d.raw\n' "$3" "$SCRATCH" "$1" "$2"
done | LANG=C parallel -j2
