# encoding: utf-8
print(2 + 2)
print(2 // 3)

# 序列测试
user = ['xws', 12, 173]
family = ['father', 'mother']
duck = [user, family]
print(duck)

# 索引 -- 访问单个数据
print("user[0] = " + user[0])

# 分片 -- 访问一组数据，第一个包含在分片里面，第二个不包含在分片里面
num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
print("step = 1,", num[3:5])
print("step = 2", num[3:10:2])
print("get index 3 after all element", num[3:])
print("get index 9 before all element", num[:9])

print("add:", [1, 2, 3] + [4, 5, 6])
print("mul:", [123] * 4)

print("x in ", 'x' in [1, 2, 3, "x"])

database = [['xws', '123'], ['xwa', '456'], ['srx', '789']]
print("test in:", ['xws', '123'] in database)

print("database.length = ", len(database))
print("user.length = ", len(user))
# print("user.max = ", max(user))
# print("user.min = ", min(user))
# print (input)
