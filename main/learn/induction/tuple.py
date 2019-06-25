# encoding: utf-8
# 元组和list一样，只是不能修改
users = (list("xws"))
print users

# 只有一个元素的元组
users = (1,)
print "only one element", users

# 多个元素的元组
users = (1, 2, 3, 4)
print "have four element", users

# 将序列转换为元组
users = tuple(list("i love you"))
print "tuple demo", users
