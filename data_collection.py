#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sys
import os
sys.path.append(os.path.abspath('..'))

import requests
import json
import csv
import pandas as pd
import adgeek_permission as permission


# In[63]:


class FBAPI:
    LIMIT = 100
    def __init__(self, account_id):
        self.account_id = account_id
        self.access_token = self.get_token()
        self.headers = self.get_headers()
        self.params = {'fields': 'creative{body}', 'limit': FBAPI.LIMIT}
        self.url = self.get_url()
    
    def get_token(self):
        access_token = permission.FacebookPermission(self.account_id).get_token()
        
        return access_token
    
    def get_headers(self):
        headers = {'Authorization': 'Bearer {}'.format(self.access_token)}
        
        return headers
    
    def get_url(self):
        url = '{}act_{}/ads'.format(permission.FACEBOOK_API_VERSION_URL, self.account_id)
        
        return url
    
    def get_raw_text(self):
        data_df = pd.DataFrame()
        url = self.url
        n = 1
        
        res = requests.get(url, headers=self.headers, params=self.params)
        while True:
            if n != 1:
                res = requests.get(url, headers=self.headers)
                
            data = json.loads(res.text)
            for d in data['data']:
                try:
                    data_df = data_df.append(d['creative'], ignore_index=True)
                except:
                    pass
            
            print('batch {} complete'.format(n))
            
            if data['paging'].get('next'):
                url = data['paging']['next']
                n += 1
            else:
                print('data collection finished')
                break

        data_df = data_df.drop('id', axis=1).drop_duplicates()
        fp = 'raw_text.csv'
        data_df.to_csv(fp, index=False)

        return data_df


# In[64]:


if __name__ == '__main__':
    account_id = 'MY_ACCOUNT_ID
    fbapi = FBAPI(account_id)
    data_df = fbapi.get_raw_text()


# In[67]:


# !jupyter nbconvert --to script preprocessing.ipynb


# In[ ]:




