# encoding: utf-8
import cmath

# 字符串格式化
# list会被格式化成一个值，只有元组和字典才能被分解
format = "i live you, %s, please believe me"
name = "xinxin"
print format % name

# 格式化浮点数
f = "P with three decimals: %.3f"
print f % cmath.pi

# 模版字符串

# 元组参与字符串格式化
print "%s plus %s = %s" % (1, 1, 2)

# 使用十进制进行转换
print "the 42 to x  %d " % (42,)

# 字段宽度和高度
print "the pi = %10f" % cmath.pi
print "the pi = %10.2f" % cmath.pi
print "the pi = %.2f" % cmath.pi
print "the pi = %.5s" % "i love you !"

# 填充
print "the pi = %010f" % cmath.pi
print "the pi = %-10f" % cmath.pi
print "the pi = %+10f" % cmath.pi
print "the pi = % 10f" % cmath.pi

# 包含0-9的字符串
str = "i love you !"

print "i index = ", str.find("i")
print "love index = ", str.find("love")

str = "+"
str = str.join(list("i love you !"))
print "after join str :", str

# 字符串小写
str = "AbCd"
print "lower ", str.lower()

# 首字母大写
str = "i love you"
print "title ", str.title()

# 替换字符串
str = str.replace("love", "fuck")
print "after replace ", str

# 去除两侧的字符串
str = " i love you ".strip()
print "after strip", str
