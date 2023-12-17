from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs
from sqlalchemy import create_engine
import pandas as pd

def get_advert_data(building_type, where):
    base_url = f'https://www.aruodas.lt/{building_type}/{where}/puslapis/{{}}/?FOrder=AddDate'
    driver = webdriver.Chrome() 
    data_list = []
    
    try:
        page = 1
        while True:
            url = base_url.format(page)
            driver.get(url)
            
            try:
                cookie_banner = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'onetrust-reject-all-handler'))
                )
                cookie_banner.click()
            except:
                pass
            
            advert_links = [link.get_attribute('href') for link in driver.find_elements(By.XPATH, '//h3/a[@href]')]
            
            if not advert_links:
                break  
            
            for link in advert_links:
                driver.get(link)
                data_dict = {}
                price_element = driver.find_element(By.CLASS_NAME, 'price-eur')
                price_value = price_element.text
                data_dict['Price (EUR)'] = price_value
                
                try:
                    wait = WebDriverWait(driver, 10)
                    a_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-type='map']")))
                    href_value = a_element.get_attribute('href')
                    query_params = parse_qs(urlparse(href_value).query)
                    coordinates = query_params.get('query', [''])[0].split(',')
                    latitude = coordinates[0]
                    longitude = coordinates[1]
                    data_dict['Latitude'] = latitude
                    data_dict['Longitude'] = longitude
                except:
                    pass
                
                dt_elements = driver.find_elements(By.TAG_NAME, 'dt')
                dd_elements = driver.find_elements(By.TAG_NAME, 'dd')
                
                for dt, dd in zip(dt_elements, dd_elements):
                    property_name = dt.text
                    property_value = dd.text
                    data_dict[property_name] = property_value
                
                data_list.append(data_dict)
                
            print(f'Scraped page {page}')
            page += 1
    
    finally:
        df = pd.DataFrame(data_list)
        driver.quit()
        return df

flats_df = get_advert_data('butai', 'vilniuje')

engine = create_engine('postgresql://xxxxxx:yyyyyyy@qqqqqqqqqqq.rds.amazonaws.com:0000/db_name')
existing_data_df = pd.read_sql('SELECT * FROM uncleaned_flats', con=engine)

combined_df = pd.concat([existing_data_df, flats_df], ignore_index=True)
unique_data_df = combined_df.drop_duplicates(subset=combined_df.columns.difference(['Price (EUR)', 'Nuoroda']), keep='last')
rows_to_insert = len(unique_data_df)

unique_data_df.to_sql('uncleaned_flats', con=engine, if_exists='replace', index=False)
print(f"Number of rows: {rows_to_insert}")
