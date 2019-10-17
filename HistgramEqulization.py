import cv2
import os

#直方圖均化
def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

#讀取資料夾的圖片名稱
BedroomPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/bedroom/"
bedroom = [f for f in os.listdir(BedroomPath) if os.path.isfile(os.path.join(BedroomPath, f))]
CoastPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/coast/"
coast = [f for f in os.listdir(CoastPath) if os.path.isfile(os.path.join(CoastPath, f))]
ForestPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/forest/"
forest = [f for f in os.listdir(ForestPath) if os.path.isfile(os.path.join(ForestPath, f))]
HighwayPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/highway/"
highway = [f for f in os.listdir(HighwayPath) if os.path.isfile(os.path.join(HighwayPath, f))]
InsidecityPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/insidecity/"
insidecity = [f for f in os.listdir(InsidecityPath) if os.path.isfile(os.path.join(InsidecityPath, f))]
KitchenPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/kitchen/"
kitchen = [f for f in os.listdir(KitchenPath) if os.path.isfile(os.path.join(KitchenPath, f))]
LivingroomPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/livingroom/"
livingroom = [f for f in os.listdir(LivingroomPath) if os.path.isfile(os.path.join(LivingroomPath, f))]
MountainPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/mountain/"
mountain = [f for f in os.listdir(MountainPath) if os.path.isfile(os.path.join(MountainPath, f))]
OfficePath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/office/"
office = [f for f in os.listdir(OfficePath) if os.path.isfile(os.path.join(OfficePath, f))]
OpencountryPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/opencountry/"
opencountry = [f for f in os.listdir(OpencountryPath) if os.path.isfile(os.path.join(OpencountryPath, f))]
StreetPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/street/"
street = [f for f in os.listdir(StreetPath) if os.path.isfile(os.path.join(StreetPath, f))]
SuburbPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/suburb/"
suburb = [f for f in os.listdir(SuburbPath) if os.path.isfile(os.path.join(SuburbPath, f))]
TallbuildingPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train/tallbuilding/"
tallbuilding = [f for f in os.listdir(TallbuildingPath) if os.path.isfile(os.path.join(TallbuildingPath, f))]

#新路徑命名
BedroomPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/bedroom/"
CoastPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/coast/"
ForestPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/forest/"
HighwayPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/highway/"
InsidecityPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/insidecity/"
KitchenPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/kitchen/"
LivingroomPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/livingroom/"
MountainPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/mountain/"
OfficePath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/office/"
OpencountryPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/opencountry/"
StreetPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/street/"
SuburbPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/suburb/"
TallbuildingPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/train2/tallbuilding/"

#Test Data 名稱讀取
TestPath = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/test2/test/"
TestPath2 = r"C:/Users/Ping/Desktop/Computer_Vision/HW1/cs-ioc5008-hw1/dataset/dataset/test3/test/"
test = [f for f in os.listdir(TestPath) if os.path.isfile(os.path.join(TestPath, f))]

#進行直方圖均化並儲存
bedroon_img = {}
for i in range(len(bedroom)):
    bedroon_img[i] = cv2.imread(BedroomPath + bedroom[i])
    bedroon_img[i] = hisEqulColor(bedroon_img[i])
    cv2.imwrite(BedroomPath2 + str(i) + '.jpg', bedroon_img[i])

coast_img = {}
for i in range(len(coast)):
    coast_img[i] = cv2.imread(CoastPath + coast[i])
    coast_img[i] = hisEqulColor(coast_img[i])
    cv2.imwrite(CoastPath2 + str(i) + '.jpg', coast_img[i])

forest_img = {}
for i in range(len(forest)):
    forest_img[i] = cv2.imread(ForestPath + forest[i])
    forest_img[i] = hisEqulColor(forest_img[i])
    cv2.imwrite(ForestPath2 + str(i) + '.jpg', forest_img[i])

highway_img = {}
for i in range(len(highway)):
    highway_img[i] = cv2.imread(HighwayPath + highway[i])
    highway_img[i] = hisEqulColor(highway_img[i])
    cv2.imwrite(HighwayPath2 + str(i) + '.jpg', highway_img[i])

insidecity_img = {}
for i in range(len(insidecity)):
    insidecity_img[i] = cv2.imread(InsidecityPath + insidecity[i])
    insidecity_img[i] = hisEqulColor(insidecity_img[i])
    cv2.imwrite(InsidecityPath2 + str(i) + '.jpg', insidecity_img[i])

kitchen_img = {}
for i in range(len(kitchen)):
    kitchen_img[i] = cv2.imread(KitchenPath + kitchen[i])
    kitchen_img[i] = hisEqulColor(kitchen_img[i])
    cv2.imwrite(KitchenPath2 + str(i) + '.jpg', kitchen_img[i])

livingroom_img = {}
for i in range(len(livingroom)):
    livingroom_img[i] = cv2.imread(LivingroomPath + livingroom[i])
    livingroom_img[i] = hisEqulColor(livingroom_img[i])
    cv2.imwrite(LivingroomPath2 + str(i) + '.jpg', livingroom_img[i])

mountain_img = {}
for i in range(len(mountain)):
    mountain_img[i] = cv2.imread(MountainPath + mountain[i])
    mountain_img[i] = hisEqulColor(mountain_img[i])
    cv2.imwrite(MountainPath2 + str(i) + '.jpg', mountain_img[i])

office_img = {}
for i in range(len(office)):
    office_img[i] = cv2.imread(OfficePath + office[i])
    office_img[i] = hisEqulColor(office_img[i])
    cv2.imwrite(OfficePath2 + str(i) + '.jpg', office_img[i])

opencountry_img = {}
for i in range(len(opencountry)):
    opencountry_img[i] = cv2.imread(OpencountryPath + opencountry[i])
    opencountry_img[i] = hisEqulColor(opencountry_img[i])
    cv2.imwrite(OpencountryPath2 + str(i) + '.jpg', opencountry_img[i])

street_img = {}
for i in range(len(street)):
    street_img[i] = cv2.imread(StreetPath + street[i])
    street_img[i] = hisEqulColor(street_img[i])
    cv2.imwrite(StreetPath2 + str(i) + '.jpg', street_img[i])

suburb_img = {}
for i in range(len(suburb)):
    suburb_img[i] = cv2.imread(SuburbPath + suburb[i])
    suburb_img[i] = hisEqulColor(suburb_img[i])
    cv2.imwrite(SuburbPath2 + str(i) + '.jpg', suburb_img[i])

tallbuilding_img = {}
for i in range(len(tallbuilding)):
    tallbuilding_img[i] = cv2.imread(TallbuildingPath + tallbuilding[i])
    tallbuilding_img[i] = hisEqulColor(tallbuilding_img[i])
    cv2.imwrite(TallbuildingPath2 + str(i) + '.jpg', tallbuilding_img[i])

test_img = {}
for i in range(len(test)):
    test_img[i] = cv2.imread(TestPath + test[i])
    test_img[i] = hisEqulColor(test_img[i])
    cv2.imwrite(TestPath2 + test[i] + '.jpg', test_img[i])