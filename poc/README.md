```
4X4_X2200_Y-77900_488_nm_2x_Left_000008.raw
4X4_X2200_Y-77900_488_nm_2x_Left_000008.raw_meta.txt
4X4_X2200_Y-77900_488_nm_2x_Right_000009.raw
4X4_X2200_Y-77900_488_nm_2x_Right_000009.raw_meta.txt
4X4_X2200_Y-83408_488_nm_2x_Left_000012.raw
4X4_X2200_Y-83408_488_nm_2x_Left_000012.raw_meta.txt
4X4_X2200_Y-83408_488_nm_2x_Right_000013.raw
4X4_X2200_Y-83408_488_nm_2x_Right_000013.raw_meta.txt
4X4_X-2808_Y-77900_488_nm_2x_Left_000000.raw
4X4_X-2808_Y-77900_488_nm_2x_Left_000000.raw_meta.txt
4X4_X-2808_Y-77900_488_nm_2x_Right_000001.raw
4X4_X-2808_Y-77900_488_nm_2x_Right_000001.raw_meta.txt
4X4_X-2808_Y-83408_488_nm_2x_Left_000004.raw
4X4_X-2808_Y-83408_488_nm_2x_Left_000004.raw_meta.txt
4X4_X-2808_Y-83408_488_nm_2x_Right_000005.raw
4X4_X-2808_Y-83408_488_nm_2x_Right_000005.raw_meta.txt
```


# stride = 4, 7.9G
scp
-j1  1m43.874s
-j2  2m6.024s
-j4  1m28.301s
-j8  1m27.277s
-j15 1m28.313s

scp -C
-j2  7m50.170s
-j4  3m59.799s
-j15 1m20.502s

rsync -z
--compress-level=1 -j2  1m28.770s
--compress-level=1 -j15 1m0.804s
--compress-level=6 -j15 1m14.325s
--compress-level=9 -j15 4m45.885s

118MB/s

# stride = 2, 63G

rsync -z
--compress-level=1 -j15 8m44.540s

120MB/s
