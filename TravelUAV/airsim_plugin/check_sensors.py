import airsim
import pprint

# 连接 25001 端口
client = airsim.MultirotorClient(port=25001)
client.confirmConnection()

print("Connected!")
print("Sensors:", client.getLidarData("Lidar1")) # 看看这句会不会报错