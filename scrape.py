from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import time

def create_driver(driver_path, user_agent):
    # Create Chrome WebDriver options
    options = Options()
    options.add_argument(f"user-agent={user_agent}")

    # Set the path to the ChromeDriver executable using the 'executable_path' option
    service = Service(driver_path)

    # Initialize the WebDriver with the specified options
    driver = webdriver.Chrome(service=service, options=options)

    return driver

def login_to_github(driver, email, password):
    # Opening GitHub's login page
    driver.get("https://github.com/login")

    # Waiting for the page to load
    time.sleep(5)

    # Entering username
    username = driver.find_element(By.ID, "login_field")
    username.send_keys(email)

    # Entering password
    pword = driver.find_element(By.ID, "password")
    pword.send_keys(password)

    # Clicking on the log in button
    driver.find_element(By.XPATH, "//input[@type='submit']").click()

    # Here, you might want to add some logic to confirm that the login was successful.
    # For example, you could check for the presence of an element that's only visible when logged in.

    return driver  # Return the driver after logging in

def main(email, password, driver_path):
    user_agent = UserAgent().random

    driver = create_driver(driver_path, user_agent)

    try:
        driver = login_to_github(driver, email, password)
        # Now 'driver' is the same WebDriver instance, but it's logged into GitHub.
        # You can continue to use 'driver' here for further tasks.
    except Exception as e:
        print(f"An error occurred: {e}")
        driver.quit()  # Only quit if an exception occurred

    # The script ends here, but the browser remains open, and 'driver' is still accessible.

    return driver  # Return the driver if you need to use it after calling main

def scrape_github_repos(driver, url):
    # Navigate to the provided URL
    driver.get(url)

    # Get the page source
    html_content = driver.page_source

    # Parse the page source through BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all h3 tags, which contain repository names
    h3_tags = soup.find_all("h3")

    # List to store repository names
    repos = []

    # Extract repository names from the h3 tags
    for h3 in h3_tags:
        a_tag = h3.find("a")  # find 'a' tags in each 'h3'
        if a_tag is not None:  # if 'a' tag is present in 'h3', get the 'href'
            href = a_tag.get('href')
            if href is not None:  # if 'href' is present in 'a', modify and add it to the list
                modified_href = href[1:]  # remove the first character, i.e., '/'
                repos.append(modified_href)

    return repos


# Use the main function
email = "your_email@example.com"
password = "your_password"
driver_path = "/path/to/your/chromedriver"

driver = main(email, password, driver_path)
# You can continue using 'driver' here if needed

# Usage of the function:
url = 'https://github.com/search?q=robotics+stars%3A%3E200&type=repositories'
repositories = scrape_github_repos(driver, url)
print(repositories)