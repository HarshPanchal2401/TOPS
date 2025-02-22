import random
import smtplib
import json
import os

class ATM:
    def __init__(self):
        self.data_file = 'atm_data.json'
        self.pin = self.get_pin()
        self.balance = self.get_balance()
        self.email = self.get_email()

    def menu(self):
        user_input = input("""
            Press 1 -> Create Pin
            Press 2 -> Change Pin
            Press 3 -> Deposit Money
            Press 4 -> Check Balance
            Press 5 -> Withdraw Money
            Press 6 -> Exit

            Enter Your Choice : """)

        if user_input == '1':
            self.create_pin()   
        elif user_input == '2':
            self.change_pin()
        elif user_input == '3':
            self.deposit() 
        elif user_input == '4':
            self.check_balance()  
        elif user_input == '5':
            self.withdraw()
        else:
            exit()

    def create_pin(self):
        email = input("Enter your Email for Verification : ")
        self.email = email
        self.save_data()
        otp = self.generate_otp(6)
        self.send_otp(otp, email)
        user_otp = input("Enter the OTP : ")

        if user_otp == otp:
            print("OTP Verified")
            if len(self.pin) == 0:
                user_pin = input("Create your Pin : ")
                self.pin = user_pin
                self.email = email
                self.save_data()
                print("Pin Created Successfully")
            else:
                print("You already created your pin.")
        else:
            print("Invalid OTP")
        self.menu()

    def change_pin(self):
        old_pin = input("Enter your old pin : ")
        if old_pin == self.pin:
            new_pin = input("Enter your new pin : ")
            if old_pin != new_pin:
                self.pin = new_pin
                self.save_data()
                print("Pin Changed Successfully")
            else:
                print("Your Old pin and New pin are the same.")
        else:
            print("Old Pin is Incorrect.")
        self.menu()

    def deposit(self):
        deposit_pin = input("Enter the Pin : ")
        if deposit_pin == self.pin:
            amount = int(input('Enter the Amount to Deposit : '))
            self.balance += amount
            self.save_data()
            print("Your Money has been Deposited Successfully.")
            self.send_notification("Credit", amount)
        else:
            print("Your Pin is Incorrect.")
        self.menu()

    def check_balance(self):
        bal_pin = input("Enter the Pin : ")
        if bal_pin == self.pin:
            print(f"Your Current Balance is {self.get_balance()}")
        else:
            print("Your Pin is Incorrect.")
        self.menu()

    def withdraw(self):
        withdraw_pin = input("Enter the Pin : ")
        if withdraw_pin == self.pin:
            try:
                amount = int(input('Enter the Withdrawal Amount  : '))
                if amount > self.balance:
                    raise ValueError
                elif self.balance == 0:
                    print("Your account is empty.")
                else:
                    self.balance -= amount
                    self.save_data()
                    print(f"Your Balance is now {self.balance}.")
                    self.send_notification("Debit", amount)
            except ValueError:
                print("Insufficient Balance.")
        else:
            print("Your Pin is Incorrect.")
        self.menu()

    def generate_otp(self, length=6):
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])

    def send_otp(self, otp, recipient_email):
        sender_email = "panchal.harsh241@gmail.com"
        sender_password = "laqv ejim wxok kisi"
        subject = "Your OTP Code"
        body = f"Your OTP is: {otp}"
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                message = f"Subject: {subject}\n\n{body}"
                server.sendmail(sender_email, recipient_email, message)
            print("OTP sent successfully!")
        except Exception as e:
            print(f"Error sending OTP: {e}")

    def send_notification(self, transaction_type, amount):
        sender_email = "panchal.harsh241@gmail.com"
        sender_password = "laqv ejim wxok kisi"
        subject = f"{transaction_type} Notification"
        body = f"Your account has been {transaction_type.lower()}ed with INR {amount}. Current balance: INR {self.balance}."
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                message = f"Subject: {subject}\n\n{body}"
                server.sendmail(sender_email, self.email, message)
            print(f"{transaction_type} notification sent successfully!")
        except Exception as e:
            print(f"Error sending {transaction_type} notification: {e}")

    def get_pin(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as file:
                data = json.load(file)
                return data.get("pin", '')
        return ''

    def get_balance(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as file:
                data = json.load(file)
                return data.get("balance", 0)
        return 0

    def get_email(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as file:
                data = json.load(file)
                return data.get("email", '')
        return ''

    def save_data(self):
        with open(self.data_file, "w") as file:
            json.dump({"pin": self.pin, "balance": self.balance, "email": self.email}, file)

user = ATM()
user.menu()

