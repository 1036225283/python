# encoding: utf-8

# super只在新式类中起作用
__metaclass__ = type


class User:
    def show(self):
        print("name is ", self.name)

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def __init__(self):
        print("this is init method,called where User()")


class Student(User):
    def __init__(self):
        super(User, self).__init__()
        print("this is student init")

    def show(self):
        print("the student name", self.name)


pass

u1 = User()
u2 = User()

u1.setName("xws")
u2.setName("sxx")

u1.show()
u2.show()

s1 = Student()
s1.setName("ssx")
s1.show()
