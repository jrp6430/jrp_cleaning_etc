from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import os
import time



def find_titles_li(user, passw, query):
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.linkedin.com/?trk=people-guest_nav-header-logo")
    # first, find the search bar and enter the query string given in the function call
    search_bar1 = driver.find_element(by=By.ID, value='session_key')
    search_bar1.send_keys(user)
    search_bar2 = driver.find_element(by=By.ID, value='session_password')
    search_bar2.send_keys(passw)
    driver.find_element(by=By.CSS_SELECTOR,
                        value="button[data-id='sign-in-form__submit-btn']").click()
    time.sleep(10)
    driver.maximize_window()
    driver.implicitly_wait(5)
    driver.find_element(by=By.CSS_SELECTOR, value="input[placeholder='Search']").send_keys(query + Keys.ENTER)
    driver.implicitly_wait(10)
    driver.find_element(by=By.XPATH, value="//button[contains(., 'People')]").click()
    driver.implicitly_wait(3)
    driver.execute_script("document.body.style.zoom='50%'")
    store = dict()
    cum_names = []
    cum_links = []
    for i in range(20):
        names, urls = get_info_from_pg(driver)
        cum_names += names
        cum_links += urls
        driver.implicitly_wait(5)
        driver.execute_script("arguments[0].click();", driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next']"))
        driver.implicitly_wait(5)
    return cum_names, cum_links


def get_info_from_pg(driver):
    page_names = driver.find_elements(by=By.CSS_SELECTOR, value="span[dir='ltr'] span[aria-hidden='true']")
    page_links = driver.find_elements(by=By.XPATH, value="//a[.//span[@dir='ltr']//span[@aria-hidden='true']]")
    names = []
    urls = []
    for i in page_names:
        name = i.get_attribute('innerHTML')
        name = name.lstrip('<!---->').rstrip('<!---->')
        names.append(name)
    for i in page_links:
        url = i.get_attribute('href')
        if "miniProfile" in url:
            urls.append(url)
    urls = list(dict.fromkeys(urls))
    return names, urls


# using python function, I want to create a series of relational tables in SQL database with the following structure
# Table 1:
    # Study Name (string)
    # Study PI (string)
    # CSV id (int or series of ints)

def read_csv_links_to_sql(process_frame):
    for i in process_frame.index:
        links = process_frame.iloc[i]['Study_CSV_Links']
        print(links)
    return

def geo_search_csv_links(query_string, organism_string):
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.ncbi.nlm.nih.gov/geo/browse/")
    # first, find the search bar and enter the query string given in the function call
    search_bar = driver.find_element(by=By.CLASS_NAME, value='searchTxt')
    search_bar.send_keys(query_string)
    # Now, that the search bar has been filled, click the submit button
    driver.find_element(by=By.CLASS_NAME, value='button').click()
    # with the search query executed, we must now expand the page size to 500 for easier access to results
    # NOTE: automatically gives most recent entries first
    page_size_select = Select(driver.find_element(by=By.ID, value='set_page_size'))
    page_size_select.select_by_value('500')
    # now, use css selector to establish filter based on host species in study
    find_species_css = '.sp[title="' + organism_string + '"]'
    driver.find_element(by=By.CSS_SELECTOR, value=find_species_css).click()
    # now, add another filter by clicking the CSV option for supplementary documents
    driver.find_element(by=By.LINK_TEXT, value='CSV').click()
    # ok with the string query and organism query, we now have a list of table values with CSV download links
    # NOTE: for the query, make it specific!
    # Methodology, cell type, specific disease
    # example I used: RNA seq cardiomyocytes - 8 experiments

    # Also, we want the study title names as dictionary keys for storing download links
    query_study_titles = driver.find_elements(by=By.CSS_SELECTOR, value='.title div')
    titles = []
    for i in query_study_titles:
        titles.append(i.get_attribute("innerHTML"))

    # Probably a good idea to add the PI to the output df too
    # for some reason the first one is always just 'Contact', so remove it after
    query_study_pis = driver.find_elements(by=By.CSS_SELECTOR, value='.contact a')
    pis = []
    for j in query_study_pis:
        pis.append(j.get_attribute('text'))
    pis.pop(0)

    # Now, we must gather the csv download links and iterate through them
    to_csv_page = driver.find_elements(by=By.CSS_SELECTOR, value='.link_icon[href*="ftp.ncbi.nlm.nih.gov"]')

    # to avoid using the back button and turning to_csv_links elements stale,
    # open the links that lead to csv download in a new tab, then parse them

    # to do this, we need to store the handle of the original tab to switch back to it
    parent = driver.window_handles[0]
    store_links = []

    for element in to_csv_page:
        # for each csv download page
        # open a new tab with ctrl + enter
        element.send_keys(Keys.CONTROL + Keys.ENTER)

        # store the window handle of the fresh tab and switch the driver to it
        child = driver.window_handles[1]
        driver.switch_to.window(child)

        # create an empty array for download links within this download page
        # also, instantiate a WebDriverWait class object.
        current_download_links = []
        wait = WebDriverWait(driver, 10)

        try:
            # using WebDriverWait, stall on new tab until all csv download links are visible
            wait.until(EC.visibility_of_all_elements_located((By.PARTIAL_LINK_TEXT, '.csv')))

            # then locate the download links with find_elements
            download_links = driver.find_elements(by=By.PARTIAL_LINK_TEXT, value='.csv')

            # use a loop to add all csv download links for the current page to an array
            for link in download_links:
                current_download_links.append(link.get_attribute('href'))

            # add array of download links to the larger cumulative link storage structure
            store_links.append(current_download_links)

            # close current tab and switch to original window with query results
            driver.close()
            driver.switch_to.window(parent)
        except:
            # if WebDriverWait fails, there is actually no .csv file download on the page
            # THEREFORE add a string to the link_store array to mark for removal
            # THEN close tab and switch back to original
            print('There was no csv file found on the download page after waiting. Close tab and move on!')
            store_links.append('No link found')
            driver.close()
            driver.switch_to.window(parent)
            continue
    driver.close()

    # We now have three arrays, one with study names, another with PI names, and a final one with csv download links
    # for each.
    # Turn these into a dataframe and output them
    d = dict()
    d['Study_Title'] = titles
    d['Study_Contact'] = pis
    d['Study_CSV_Links'] = store_links
    frame = pd.DataFrame(d)

    # remove studies from output data structure where a csv download link was not found
    drop_studies = frame[frame['Study_CSV_Links'] == 'No link found'].index
    frame.drop(drop_studies, axis=0, inplace=True)
    frame.reset_index(inplace=True, drop=True)

    return frame

