import math
import time
from PIL import Image
import numpy as np
from queue import PriorityQueue
import sys
import copy


def main():
    start_time = time.time()
    imageFile = sys.argv[1]
    image = Image.open(imageFile)
    season = sys.argv[4].lower()
    elevationFile = sys.argv[2]
    elevation = []
    with open(elevationFile, 'r') as f:
        line = f.readline()
        while line:
            temp = [float(x) for x in line.split()][:-5]
            elevation.append(temp)
            line = f.readline()
    if season == 'summer':
        terrain = np.array(Image.open(imageFile).convert('RGB')).tolist()
        speedMap = {(248, 148, 18): 6.2, (255, 192, 0): 3.1, (255, 255, 255): 4.34, (2, 208, 60): 2.48,
                    (2, 136, 40): 1.86,
                    (5, 73, 24): 0.62, (0, 0, 255): 1.24, (71, 51, 3): 8.9, (0, 0, 0): 7.15,
                    (205, 0, 101): 0.0000000000000000000000001}
        output = np.array(Image.open(imageFile).convert('RGB')).tolist()
    elif season == 'winter':
        terrain = np.array(Image.open(imageFile).convert('RGB')).tolist()
        terrain = makeIce(terrain)
        output = copy.deepcopy(terrain)
        speedMap = {(248, 148, 18): 6.2, (255, 192, 0): 3.1, (255, 255, 255): 4.34, (2, 208, 60): 2.48,
                    (2, 136, 40): 1.86,
                    (5, 73, 24): 0.62, (0, 0, 255): 0.62, (71, 51, 3): 8.9, (0, 0, 0): 7.15,
                    (205, 0, 101): 0.0000000000000000000000001, (0, 128, 128): 2.48}
    elif season == 'spring':
        terrain = np.array(Image.open(imageFile).convert('RGB')).tolist()
        terrain = makeMud(terrain, elevation)
        output = copy.deepcopy(terrain)
        speedMap = {(248, 148, 18): 6.2, (255, 192, 0): 3.1, (255, 255, 255): 4.34, (2, 208, 60): 2.48,
                    (2, 136, 40): 1.86,
                    (5, 73, 24): 0.62, (0, 0, 255): 0.62, (71, 51, 3): 8.9, (0, 0, 0): 7.15,
                    (205, 0, 101): 0.0000000000000000000000001,(139, 69, 19) : 2.48}
    elif season == 'fall':
        terrain = np.array(Image.open(imageFile).convert('RGB')).tolist()
        output = np.array(Image.open(imageFile).convert('RGB')).tolist()
        speedMap = {(248, 148, 18): 6.2, (255, 192, 0): 3.1, (255, 255, 255): 2.48, (2, 208, 60): 2.48,
                    (2, 136, 40): 1.86,
                    (5, 73, 24): 0.62, (0, 0, 255): 0.62, (71, 51, 3): 8.9, (0, 0, 0): 7.15,
                    (205, 0, 101): 0.0000000000000000000000001, (0, 128, 128): 1.86}

    maxSpeed = 8.9
    controlPoints = []
    controlPointsFile = sys.argv[3]
    with open(controlPointsFile, 'r') as f:
        line = f.readline()
        while line:
            temp = [int(x) for x in line.split()]
            controlPoints.append(temp)
            line = f.readline()

    totalDistance = 0
    for i in range(0, len(controlPoints) - 1):
        reversed_list1 = controlPoints[i][::-1]
        reversed_list2 = controlPoints[i + 1][::-1]
        pathMap = aStar(terrain, speedMap, maxSpeed, reversed_list1, reversed_list2, elevation)
        finalDestination = reversed_list2
        shortestPath = pathMap.get(tuple(finalDestination))
        for x in shortestPath:
            if x[1] == finalDestination[0] and x[2] == finalDestination[1]:
                totalDistance += x[4]
            output[x[1] - 1][x[2] - 1] = [139, 0, 0]
    print(time.time() - start_time)
    print("Total Distance : ", end='')
    print(totalDistance)
    pixels_out = []
    for row in output:
        for tup in row:
            pixels_out.append(tuple(tup))
    image_out = Image.new(image.mode, image.size)
    image_out.putdata(pixels_out)
    outputName = sys.argv[5]
    image_out.save(outputName)


def aStar(terrain, speedMap, maxSpeed, source, destination, elevation):
    pathMap = {}
    pq = PriorityQueue()
    result = set()
    horizontal = 10.29
    vertical = 7.55
    diagonal = 12.76
    pq.put((calHeuristic(maxSpeed, source, destination), source[0], source[1], 0, 0, None))
    pathMap[tuple(source)] = [[calHeuristic(maxSpeed, source, destination), source[0], source[1], 0, 0, None]]
    while not pq.empty():
        currNode = pq.get()
        currPoint = tuple(currNode[1:3])
        if currNode[1] == destination[0] and currNode[2] == destination[1]:
            tempList = pathMap.get(currNode[5])[:]
            tempList.append(currNode)
            pathMap[tuple(currPoint)] = tempList
            return pathMap
        if currPoint not in result:
            if currPoint not in pathMap:
                tempList = pathMap.get(currNode[5])[:]
                tempList.append(currNode)
                pathMap[tuple(currPoint)] = tempList

            result.add(currPoint)
            addHorizontal(pq, currNode, speedMap, elevation, horizontal, terrain, pathMap, maxSpeed, destination,
                          result)
            addVertical(pq, currNode, speedMap, elevation, vertical, terrain, pathMap, maxSpeed, destination, result)
            addDiagonals(pq, currNode, speedMap, elevation, diagonal, terrain, pathMap, maxSpeed, destination, result)
    return pathMap


def addHorizontal(pq, currNode, speedMap, elevation, dist, terrain, pathMap, maxSpeed, destination, result):
    point1 = list(currNode)[1:3]
    parent = tuple(point1)
    if point1[0] + 1 <= 500:
        rightPoint = [point1[0] + 1, point1[1]]
        rightDist, rightCost, rightTCost = calCost(point1, rightPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                                   destination, currNode[3], currNode[4])
        rightNode = (rightTCost, rightPoint[0], rightPoint[1], rightCost, rightDist, parent)
        if tuple(rightPoint) not in result:
            pq.put(rightNode)

    if point1[0] - 1 >= 1:
        leftPoint = [point1[0] - 1, point1[1]]
        leftDist, leftCost, leftTCost = calCost(point1, leftPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                                destination, currNode[3],
                                                currNode[4])
        leftNode = (leftTCost, leftPoint[0], leftPoint[1], leftCost, leftDist, parent)
        if tuple(leftPoint) not in result:
            pq.put(leftNode)


def addVertical(pq, currNode, speedMap, elevation, dist, terrain, pathMap, maxSpeed, destination, result):
    point1 = list(currNode)[1:3]
    parent = tuple(point1)
    if point1[1] + 1 <= 395:
        topPoint = [point1[0], point1[1] + 1]
        topDist, topCost, topTCost = calCost(point1, topPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                             destination, currNode[3],
                                             currNode[4])
        topNode = (topTCost, topPoint[0], topPoint[1], topCost, topDist, parent)
        if tuple(topPoint) not in result:
            pq.put(topNode)
    else:
        print("out of bounds")

    if point1[1] - 1 >= 1:
        bottomPoint = [point1[0], point1[1] - 1]
        bottomDist, bottomCost, bottomTCost = calCost(point1, bottomPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                                      destination, currNode[3],
                                                      currNode[4])

        bottomNode = (bottomTCost, bottomPoint[0], bottomPoint[1], bottomCost, bottomDist, parent)
        if tuple(bottomPoint) not in result:
            pq.put(bottomNode)
    else:
        print("out of bounds")


def addDiagonals(pq, currNode, speedMap, elevation, dist, terrain, pathMap, maxSpeed, destination, result):
    point1 = list(currNode)[1:3]
    parent = tuple(point1)
    if point1[0] + 1 <= 500 and point1[1] + 1 <= 395:
        topRightPoint = [point1[0] + 1, point1[1] + 1]
        trDist, topRightCost, trTCost = calCost(point1, topRightPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                                destination, currNode[3],
                                                currNode[4])
        trNode = (trTCost, topRightPoint[0], topRightPoint[1], topRightCost, trDist, parent)
        if tuple(topRightPoint) not in result:
            pq.put(trNode)
    else:
        print("out of bounds")

    if point1[0] + 1 <= 500 and point1[1] - 1 >= 1:
        bottomRightPoint = [point1[0] + 1, point1[1] - 1]
        brDist, bottomRightCost, brTCost = calCost(point1, bottomRightPoint, speedMap, elevation, dist, terrain,
                                                   maxSpeed,
                                                   destination, currNode[3],
                                                   currNode[4])
        brNode = (brTCost, bottomRightPoint[0], bottomRightPoint[1], bottomRightCost, brDist, parent)
        if tuple(bottomRightPoint) not in result:
            pq.put(brNode)
    else:
        print("out of bounds")

    if point1[0] - 1 >= 1 and point1[1] + 1 <= 395:
        topLeftPoint = [point1[0] - 1, point1[1] + 1]
        tlDist, topLeftCost, tlTCost = calCost(point1, topLeftPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                               destination, currNode[3],
                                               currNode[4])
        tlNode = (tlTCost, topLeftPoint[0], topLeftPoint[1], topLeftCost, tlDist, parent)
        if tuple(topLeftPoint) not in result:
            pq.put(tlNode)
    else:
        print("out of bounds")

    if point1[0] - 1 >= 1 and point1[1] - 1 >= 1:
        bottomLeftPoint = [point1[0] - 1, point1[1] - 1]
        blDist, bottomLeftCost, blTCost = calCost(point1, bottomLeftPoint, speedMap, elevation, dist, terrain, maxSpeed,
                                                  destination, currNode[3],
                                                  currNode[4])
        blNode = (blTCost, bottomLeftPoint[0], bottomLeftPoint[1], bottomLeftCost, blDist, parent)
        if tuple(bottomLeftPoint) not in result:
            pq.put(blNode)
    else:
        print("out of bounds")


def calCost(point1, point2, speedMap, elevation, dist, terrain, maxSpeed, destination, prevCost, prevDist):
    # reversed_list = os[::-1]
    # reversed_list = os[::-1]

    slope = abs((elevation[point2[0] - 1][point2[1] - 1] - elevation[point1[0] - 1][point1[1] - 1])) / dist
    cosTheta = 1 / (math.sqrt(1 + slope * slope))
    newDist = (dist / cosTheta)
    newCost = prevCost + (newDist / speedMap.get(tuple(terrain[point2[0] - 1][point2[1] - 1])))
    tCost = newCost + calHeuristic(maxSpeed, point2, destination)
    return newDist + prevDist, newCost, tCost


def calHeuristic(maxSpeed, point1, destination):
    return calDist(point1, destination) / maxSpeed


def calDist(point1, point2):
    return math.sqrt((((point2[1] - point1[1]) * 7.55) ** 2) + ((point2[0] - point1[0]) * 10.29) ** 2)


def bfsSpring(terrain, i, j, elevation):
    q = [(i, j)]
    elevationStart = elevation[i][j]
    visited = [[0] * len(terrain[0])] * len(terrain)
    level = 1
    while level <= 14:
        tempList = q[:]
        while len(tempList) != 0:
            tempTuple = tempList.pop(0)
            if visited[tempTuple[0]][tempTuple[1]] == 1:
                continue
            visited[i][j] = 1
            i = tempTuple[0]
            j = tempTuple[1]
            if i + 1 < 500 and terrain[i + 1][j] != [0, 0, 255] and visited[i + 1][j] == 0 \
                    and -elevationStart + elevation[i + 1][j] < 1:
                q.append((i + 1, j))
            if i - 1 >= 0 and terrain[i - 1][j] != [0, 0, 255] and visited[i - 1][j] == 0 \
                    and -elevationStart + elevation[i - 1][j] < 1:
                q.append((i - 1, j))
            if j + 1 < 395 and terrain[i][j + 1] != [0, 0, 255] and visited[i][j + 1] == 0 \
                    and -elevationStart + elevation[i][j+1] < 1:
                q.append((i, j + 1))
            if j - 1 >= 0 and terrain[i][j - 1] != [0, 0, 255] and visited[i][j - 1] == 0 \
                    and -elevationStart + elevation[i][j-1] < 1:
                q.append((i, j - 1))
            if i + 1 < 500 and j + 1 < 395 and terrain[i + 1][j + 1] != [0, 0, 255] and visited[i + 1][j + 1] == 0\
                    and -elevationStart + elevation[i + 1][j+1] < 1:
                q.append((i + 1, j + 1))
            if i + 1 < 500 and j - 1 >= 0 and terrain[i + 1][j - 1] != [0, 0, 255] and visited[i + 1][j - 1] == 0\
                    and -elevationStart + elevation[i + 1][j-1] < 1:
                q.append((i + 1, j - 1))
            if i - 1 >= 0 and j + 1 < 395 and terrain[i - 1][j + 1] != [0, 0, 255] and visited[i - 1][j + 1] == 0\
                    and -elevationStart + elevation[i - 1][j+1] < 1:
                q.append((i - 1, j + 1))
            if i - 1 >= 0 and j - 1 < 395 and terrain[i - 1][j - 1] != [0, 0, 255] and visited[i - 1][j - 1] == 0\
                    and -elevationStart + elevation[i - 1][j-1] < 1:
                q.append((i - 1, j - 1))
        level += 1
    return q


def makeMud(terrain, elevation):
    l = []
    for i in range(1, len(terrain) - 1):
        for j in range(1, len(terrain[0]) - 1):
            if terrain[i][j] == [0, 0, 255]:
                l.append((i, j))

    for t in l:
        q = bfsSpring(terrain, t[0], t[1], elevation)
        if len(q) != 1:
            for x in q:
                terrain[x[0]][x[1]] = [139, 69, 19]

    return terrain


def makeIce(terrain):
    l = []
    for i in range(1, len(terrain) - 1):
        for j in range(1, len(terrain[0]) - 1):
            if terrain[i][j] == [0, 0, 255] and (terrain[i + 1][j] != [0, 0, 255] or terrain[i - 1][j] != [0, 0, 255] or
                                                 terrain[i][j + 1] != [0, 0, 255] or terrain[i][j - 1] != [0, 0, 255] or
                                                 terrain[i + 1][j + 1] != [0, 0, 255] or
                                                 terrain[i + 1][j - 1] != [0, 0, 255] or
                                                 terrain[i - 1][j + 1] != [0, 0, 255] or
                                                 terrain[i - 1][j - 1] != [0, 0, 255]):
                l.append((i, j))

    for t in l:
        q = bfs(terrain, t[0], t[1])
        for x in q:
            terrain[x[0]][x[1]] = [0, 128, 128]

    return terrain


def bfs(terrain, i, j):
    q = [(i, j)]
    visited = [[0] * len(terrain[0])] * len(terrain)
    level = 1
    while level <= 7:
        tempList = q[:]
        while len(tempList) != 0:
            tempTuple = tempList.pop(0)
            if visited[tempTuple[0]][tempTuple[1]] == 1:
                continue
            visited[i][j] = 1
            i = tempTuple[0]
            j = tempTuple[1]
            if i + 1 < 500 and terrain[i + 1][j] == [0, 0, 255] and visited[i + 1][j] == 0:
                q.append((i + 1, j))
            if i - 1 >= 0 and terrain[i - 1][j] == [0, 0, 255] and visited[i - 1][j] == 0:
                q.append((i - 1, j))
            if j + 1 < 395 and terrain[i][j + 1] == [0, 0, 255] and visited[i][j + 1] == 0:
                q.append((i, j + 1))
            if j - 1 >= 0 and terrain[i][j - 1] == [0, 0, 255] and visited[i][j - 1] == 0:
                q.append((i, j - 1))
            if i + 1 < 500 and j + 1 < 395 and terrain[i + 1][j + 1] == [0, 0, 255] and visited[i + 1][j + 1] == 0:
                q.append((i + 1, j + 1))
            if i + 1 < 500 and j - 1 >= 0 and terrain[i + 1][j - 1] == [0, 0, 255] and visited[i + 1][j - 1] == 0:
                q.append((i + 1, j - 1))
            if i - 1 >= 0 and j + 1 < 395 and terrain[i - 1][j + 1] == [0, 0, 255] and visited[i - 1][j + 1] == 0:
                q.append((i - 1, j + 1))
            if i - 1 >= 0 and j - 1 < 395 and terrain[i - 1][j - 1] == [0, 0, 255] and visited[i - 1][j - 1] == 0:
                q.append((i - 1, j - 1))
        level += 1
    return q


main()
