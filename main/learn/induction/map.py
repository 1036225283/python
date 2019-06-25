# encoding: utf-8
from copy import deepcopy

user = {"name": "xws", "age": 12, "sex": "男"}
print user

# 创建map
user = {}
user["name"] = "xws"
print user

# map的格式化输出
print "my name is %(name)s" % user

# 清除所有的key and value
user1 = user
user.clear()
print "after clear user", user
print "after clear user1", user1

# 浅拷贝
user = {"name": "xws", "age": 12, "sex": "男"}
user1 = user.copy()
print "after copy user", user1

# 深拷贝
user1 = deepcopy(user)
print "after deep copy", user1

# 创建空值的map
no = user.fromkeys(["name", "age", "sex"])
print "空值", no

# 访问map的值
print "user.name = ", user.get("name")

print "items", user.items()

# 返回所有的键
print "all key ", user.keys()

# 移除元素
user.pop("name")
print "after pop name", user

# 随机移除
user.popitem()
print "after random pop", user

# 设置默认值
user.setdefault("name", "xws")
print "after default value", user

user.setdefault("name", "sxx")
print "after default value", user

# 更新操作
user1 = {"name": "sxx"}
user1.update(user)
print "after update from user", user1

# value集合
print "user.values = ", user.values()

# dict函数
user = dict(name="sxx", age=34)
print "after create map from dict", user
