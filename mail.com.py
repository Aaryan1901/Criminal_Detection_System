import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(sender_email, sender_password, recipient_email, subject, body):
    try:
        # Set up the server
        smtp_server = 'smtp.gmail.com'  # For Gmail
        smtp_port = 587  # For TLS

        # Create a MIME object
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the body with the msg instance
        msg.attach(MIMEText(body, 'plain'))

        # Create a secure SSL context and log in to the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, msg.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        # Terminate the session
        server.quit()


# Example usage
if __name__ == "__main__":
    sender_email = "aaryan.m299@ptuniv.edu.in"
    sender_password = "pvlm nqrj ojkx rxvv"
    recipient_email = "mohanaryan21@gmail.com"
    subject = "Test Email"
    body = "This is a test email sent from a Python script."

    send_email(sender_email, sender_password, recipient_email, subject, body)
