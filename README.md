# Automatic_stitcing

## 각 함수에 대한 설명 
 

- `ransac(points1, points2, max_iters=1000, threshold=4.0)`: 
이 함수는 RANSAC 알고리즘을 구현한 것으로 points1과 points2는 두 이미지 간에 대응되는 키포인트들의 좌표를 나타내는 배열입니다. 
해당 function에서 RANSAC은 무작위로 선택한 4개의 포인트 집합에 대한 homography 행렬을 계산하고 최상의 결과를 찾을 때까지 여러 번 반복하고 threshold는 호모그래피 행렬을 적용한 결과와 실제 대응점 간의 거리가 이 값보다 작아야 하는 임계값으로 이 값 이내에 있는 대응점만이 in-lier로써 homography 행렬을 추정하는데 사용됩니다. 

 

- `warp_image(image, homography, output_size)`: 
  이 함수는 homography 행렬을 사용하여 이미지를 warp하고 panorama로 병합하는 역할을 합니다. 현재 코드에서는 image1 -> image2에 대응되는 homography를 구하므로 image2를 warping한 뒤에 image1에 덧붙이는 형태로 panorama image를 구현하였습니다. 

## 전체 코드의 흐름 

Panorama image를 구하기 위하여 cv2의 ORB function을 이용하여 image1과 image2의 keypoint들과 descriptor들을 구합니다.  
이 후 bfmatch를 사용하여 image1과 image2에 대한 discriptor를 비교하고 이를 각 point들 간의 거리(=유사도) 순서로 정렬한 뒤 위에서 설명한 Ransac함수와 warp_image 함수를 이용하여 panorama image를 구할 수 있습니다. 

 
