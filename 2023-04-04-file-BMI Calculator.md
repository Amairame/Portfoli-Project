```python
# BMI CALCULATOR
```


```python

name =input("Enter your name: ")

weight =int(input("Enter your weight in pounds: ")) 

height = int(input("Enter your height in inches: "))

BMI = (weight * 703) / (height * height)

print(BMI)

if BMI >0:
    if(BMI<18.5):
        print(name +", you are Underweight. You need to increase your food intake")
    elif (BMI <= 24.9):
        print(name + ", you have a normal weight.")
    elif (BMI < 29.9):
        print(name + ", you are Overweight. You need to decrease your food intake a little bit")
    elif (BMI < 34.9):
        print(name + ", you are Obese. You need to work out a few times a week")
    elif (BMI < 39.9):
        print(name +", you are Severely Obese. You need to decrease your food intake and start working out ")
    else:
        print(name + ", you are Morbidly Obese. You need to talk to  doctor and start a healthy diet")
else:
    print("Enter Valid Input")


```

    Enter your name:  MariamaEnter your weight in pounds:  122Enter your height in inches:  6123.049180327868854
    Mariama, you have a normal weight.

