# encoding: utf-8


# 创建自己的异常
class SomeException(Exception):
    pass


# 可以用元组捕捉多个异常
try:
    raise Exception("hehe")
except Exception, e:
    print "this is exception by zero"
finally:
    print "this is finally"


def __init__():
    print "this is init"
