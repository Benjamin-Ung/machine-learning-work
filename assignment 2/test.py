import math
points = [  
    [220, 20, 60, 1] ,
    [255, 99, 21, 1],
    [250, 128, 14, 1],
    [144, 238, 144, 2],
    [107, 142, 35, 2 ],
    [46, 139, 87, 2 ],
    [64, 224, 208, 3 ],
    [176, 224, 23, 3 ],
    [100, 149, 237, 3 ],
    [154, 205, 50, -1]
    ]
for i in range(len(points)):
    print(math.sqrt((points[9][0] - points[i][0])**2 + (points[9][1] - points[i][1])**2 + (points[9][2] - points[i][2])**2 )) 