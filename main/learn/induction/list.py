# encoding: utf-8

users = list("hello world")
print "users = ", users

users[2] = 2
print "list[2] = 2", users

del users[0]
print "del users[0]", users

# 分片赋值
users[5:] = [21, 22, 23, 24, 25]
print "add = ", users

# 插入新的list
users[1:1] = list("i love you !")
print "插入 i love you ! 后，新的users", users

# 删除某个分片
users[1:3] = []
print "删除后的分片：", users
# 列表方法

# 追加元素
users.append("123")
print "users.append = ", users

# 统计某个元素出现的次数
print "l.count in users = ", users.count('l')

# 扩展列表
users.extend(list("welcome to chain !"))
print "users.extend = ", users

# 找到某项元素的索引
print "users.indexOf('welcome') = ", users.index(21)

# 插入元素
users.insert(0, "fuck")
print "after insert", users

# 移除最后一个元素
users.pop()
print "after pop last element", users

# 移除第一个元素
print "after pop first element", users

# 移除某项元素第一次出现的元素
users.remove('e')
print "after remove first e", users

# 反向逆序操作
users.reverse()
print "after reverse", users

tmp = users[:]
tmp.sort(reverse=True)
# 排序
print "after sort tmp", tmp

# users.sort(key=len)
print users
