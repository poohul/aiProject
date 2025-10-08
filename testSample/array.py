





array = [1,2,3,4,5,6]
addtemp = []
# print(array)
ppaa = "aafbc"

def callPrint():
    print(ppaa)

if __name__ == "__main__":
    for i in range (0 , len(array)):
        addtemp.insert(len(addtemp),array[i]*2)
        # addtemp[array[i]*2]
        # print(array[i])
        # callPrint()
        print(addtemp)

    array = array+addtemp
    # array.insert(len(array),addtemp)

    print(array)