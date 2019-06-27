# encoding: utf-8

user = {"name": "xws", "age": 34}

if user["name"] == "xws":
    print("this is true")
else:
    print("this is false")

if 0 < user["age"] <= 100:
    print("年龄在0～100岁之间")

# 对象比较
user1 = user
print("user = user1", user == user1)

# 引用比较
print("user is user1", user is user1)

# while循环
x = 10
while x >= 0:
    print(x)
    x = x - 1

# for循环
print("for --------------------")
l = list("hello world")
for i in l:
    print(i)

# 循环遍历map
user = {"name": "xws", "age": 12}
for key, val in user.items():
    print("key = %s,value = %s" % (key, val))

for key in user:
    print("key = %s,value = %s" % (key, user[key]))
