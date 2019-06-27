# encoding: utf-8

def test():
    print("this is func")


def add(a, b):
    "this is doc"
    print("%s + %s = %s" % (a, b, a + b))


def try_to_change(n):
    n = "change"
    print("this is ", n)


def try_to_change_user(user):
    user["name"] = "sxx"
    print("user.name = ", user["name"])


test()
add(1, 2)
n = "test"
try_to_change(n)

print("n = ", n)

user = {"name": "xws"}
try_to_change_user(user)
print("old user.name = ", user["name"])


# 关键字参数和默认值
def hello1(name, sex):
    print(" %s is %s" % (name, sex))


def hello2(sex, name):
    print(" %s is %s" % (name, sex))


hello1("xws", "boy")
hello2("boy", "xws")

hello1(name="xws", sex="boy")
hello2(name="xws", sex="boy")
hello1(sex="boy", name="xws")
hello2(sex="boy", name="xws")


# 关键字参数可以提供默认值
def hello3(name="xws", sex="boy"):
    print("hello3 %s is %s" % (name, sex))


hello3()


# 任意多参数
def hello4(*str):
    print
    str


hello4("i")
hello4("i", "love", "you")


# 参数收集
def hello5(name, *other, **obj):
    print
    name
    print
    other
    print
    obj


hello5("xws", 21, 173, 80, sex="boy", age=12)
print
"test************************"
# 参数收集逆过程
l = [21, 173, 80]
hello5("xws", *l, sex="boy", age=12)

# 作用域
scope = vars()
print("scope = ", scope)


# 全局变量字典

def hello6():
    user = "user"
    print(locals()["user"])
    print(globals()["l"])
    global l
    l[0] = 32
    print(globals()["l"])


# 局部变量字典

hello6()
