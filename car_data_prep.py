import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency

def prepare_data(dataset) -> pd.DataFrame:
    print("Starting data preparation...")
    
    df = dataset.copy()
    print("Data copied.")
    
    # הורדת כפילויות
    dataset = df.drop_duplicates().copy()
    print("Duplicates removed.")
    
    # סידור ערכים חסרים
    dataset.replace(['None', 'לא מוגדר', None], np.nan, inplace=True)
    print("Missing values handled.")
    
    # סידור יצרן
    dataset['manufactor'].replace('Lexsus', 'לקסוס', inplace=True)
    print("Manufacturer names fixed.")
    
    # סידור דגמים
    dataset['model'] = dataset['model'].str.strip()
    dataset['model'] = dataset['model'].str.replace(' החדשה| חדשה', '', regex=True)
    print("Model names fixed.")
    
    for i in range(len(dataset)):
        if pd.notna(dataset.iloc[i, 0]) and pd.notna(dataset.iloc[i, 2]) and dataset.iloc[i, 0] in dataset.iloc[i, 2]:
            dataset.iloc[i, 2] = dataset.iloc[i, 2].replace(dataset.iloc[i, 0], '')
    print("Model names cleaned.")
    
    # סידור דגמים ספציפיים לפי יצרן
    def replace_if_contains(value, substring, replacement):
        if isinstance(value, str) and substring in value:
            return replacement
        return value

    dataset['model'] = dataset['model'].apply(replace_if_contains, args=('סיוויק', 'CIVIC'))
    dataset['model'] = np.where(np.isin(dataset['model'], ["JAZZ", "ג'אז", 'ג`אז']), 'JAZZ', dataset['model'])
    dataset['model'] = np.where(np.isin(dataset['model'], ['אקורד']), 'ACCORD', dataset['model'])
    print("Specific models fixed.")
    
    for i in range(len(dataset)):
        if dataset.iloc[i, 0] == 'סקודה':
            s = dataset.iloc[i, 2].split()
            if len(s) > 1:
                dataset.iloc[i, 2] = s[0]
    print("Skoda models cleaned.")
    
    dataset['model'] = dataset['model'].apply(replace_if_contains, args=('קליאו', 'קליאו'))
    print("Clio models fixed.")
    
    for i in range(len(dataset)):
        if dataset.iloc[i, 0] == 'מיני':
            if dataset.iloc[i, 2] == 'ONE':
                dataset.iloc[i, 2] = 'one'
            if dataset.iloc[i, 2] == 'קאונטרימן':
                dataset.iloc[i, 2] = 'קאנטרימן'
    print("Mini models fixed.")
    
    for i in range(len(dataset)):
        if dataset.iloc[i, 0] == 'לקסוס':
            if dataset.iloc[i, 2] in ['IS300h', ' IS300H', ' IS300h']:
                dataset.iloc[i, 2] = 'IS300H'
    print("Lexus models fixed.")
    
    dataset['model'] = np.where(np.isin(dataset['model'], ['גראנד, וויאג\'ר', 'גראנד, וויאג`ר', 'וויאג`ר']), 'גראנד וויאגר', dataset['model'])
    dataset['model'] = np.where(np.isin(dataset['model'], ['ג`טה', "ג'טה"]), 'ג\'טה', dataset['model'])
    print("Grand Voyager and Jetta models fixed.")
    
    golf_variations = ['גולף', 'גולף פלוס']
    dataset['model'] = np.where(np.isin(dataset['model'], golf_variations), 'גולף', dataset['model'])
    print("Golf models fixed.")
    
    def unify_models(model):
        model = model.strip()
        if 'קופה' in model:
            return 'C-CLASS קופה'
        elif 'Taxi' in model:
            return 'C-Class'
        elif model.startswith('S-Class'):
            return 'S-Class'
        elif model.startswith('E- CLASS'):
            return 'E-Class'
        elif model.startswith('C-Class'):
            return 'C-Class'
        else:
            return model
    
    dataset['model'] = dataset['model'].apply(unify_models)
    print("Mercedes models unified.")
    
    dataset['model'] = dataset['model'].apply(replace_if_contains, args=('לנסר הדור החדש', 'לנסר'))
    dataset['model'] = dataset['model'].apply(replace_if_contains, args=('סיד', 'XCEED'))
    print("Lancer and Ceed models fixed.")
    
    for i in range(len(dataset)):
        if dataset.iloc[i, 0] == "סוזוקי" and "קרוסאובר" in dataset.iloc[i, 2]:
            dataset.iloc[i, 2] = "קרוסאובר"
    print("Suzuki models fixed.")
    
    dataset['model'] = dataset['model'].apply(replace_if_contains, args=('אונסיס', 'אוונסיס'))
    print("Avensis models fixed.")
    
    # המרת סוגי נתונים וערכים חסרים בעמודות מספריות
    print("Converting Year to numeric...")
    print("Year values before conversion:", dataset['Year'].tolist())
    dataset['Year'] = pd.to_numeric(dataset['Year'], errors='coerce')
    print("Year conversion done. Unique values:", dataset['Year'].unique())
    
    print("Converting Hand to numeric...")
    dataset['Hand'] = pd.to_numeric(dataset['Hand'], errors='coerce')
    print("Hand conversion done. Unique values:", dataset['Hand'].unique())
    
    print("Converting capacity_Engine to numeric...")
    dataset['capacity_Engine'] = pd.to_numeric(dataset['capacity_Engine'].str.replace(',', ''), errors='coerce')
    print("capacity_Engine conversion done. Unique values:", dataset['capacity_Engine'].unique())
    
    print("Converting Km to numeric...")
    dataset['Km'] = pd.to_numeric(dataset['Km'].str.replace(',', ''), errors='coerce')
    print("Km conversion done. Unique values:", dataset['Km'].unique())
    
    print("Converting Pic_num to numeric...")
    
    
    dataset['Gear'] = dataset['Gear'].apply(replace_if_contains, args=('אוטומט', 'אוטומטית'))
    
    for i in range(len(dataset)):
        if dataset.iloc[i, 6] == 'היבריד':
            dataset.iloc[i, 6] = 'היברידי'
        if dataset.iloc[i, 6] == 'טורבו דיזל':
            dataset.iloc[i, 6] = 'דיזל'
    print("Gear and Engine type fixed.")
    
    # סידור עמודות נוספות
    def standardize_city_area(column):
        for i in range(len(dataset)):
            try:
                if 'פתח' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'פתח תקווה'
                if 'פרדס' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'פרדס חנה כרכור'
                if 'חיפה' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'חיפה'
                if 'מודיעין' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'מודיעין'
                if 'ראש' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'ראש העין'
                if 'נס' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'נס ציונה'
                if 'רמלה' in dataset.iloc[i, column]:
                    dataset.iloc[i, column] = 'רמלה-לוד'
            except:
                continue

    standardize_city_area(9)  # Area column
    standardize_city_area(10) # City column
    print("City and Area standardized.")
    
    # טיפול בערכים חריגים וחסרים בעמודות מספריות
    def prepare_data1(dataset):
        dataset = dataset.dropna(subset=['Year', 'Km'])
        X = dataset[['Year']]
        y = dataset['Km']
        reg = LinearRegression().fit(X, y)
        dataset['km_predict'] = reg.predict(X)
        
        def calculate_confidence_interval(row, dataset, std_multiplier=1.96):
            year = row['Year']
            std = dataset[dataset['Year'] == year]['Km'].std()
            lower_confidence_interval = max(row['km_predict'] - std_multiplier * std, 0)
            upper_confidence_interval = row['km_predict'] + std_multiplier * std
            return pd.Series([lower_confidence_interval, upper_confidence_interval], index=['lower_confidence_interval', 'upper_confidence_interval'])
        
        dataset[['lower_confidence_interval', 'upper_confidence_interval']] = dataset.apply(calculate_confidence_interval, axis=1, dataset=dataset)
        dataset['Km'] = np.where((dataset['Km'] < dataset['lower_confidence_interval']) | (dataset['Km'] > dataset['upper_confidence_interval']),
                                 dataset.groupby('Year')['Km'].transform('median'),
                                 dataset['Km'])
        dataset['Km'] = dataset['Km'].fillna(dataset['km_predict'])
        dataset = dataset.drop(columns=['km_predict', 'lower_confidence_interval', 'upper_confidence_interval'])
        return dataset

    dataset = prepare_data1(dataset)
    print("Km outliers handled.")
    
    # טיפול בערכים חסרים בעמודות קטגוריאליות
    def chi_square_test_all_pairs(df):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        results = []
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                results.append((col1, col2, chi2, p))
        return pd.DataFrame(results, columns=['Column 1', 'Column 2', 'Chi2 Statistic', 'P-value'])

    df = dataset[['Gear', 'model']]
    chi_square_results = chi_square_test_all_pairs(df)
    print("Chi-square test for Gear done.")
    
    pivot_table = dataset.pivot_table(index='model', columns='Gear', aggfunc='size', fill_value=0).reset_index()
    
    # נבצע את החישוב רק על עמודות מספריות
    numeric_cols = pivot_table.columns[1:]
    pivot_table['Difference'] = pivot_table[numeric_cols].sum(axis=1) - pivot_table[numeric_cols].max(axis=1)
    print("Pivot table for Gear created.")
    
    for i, row in dataset.iterrows():
        if pd.isnull(row['Gear']):
            model = row['model']
            max_category = pivot_table.loc[pivot_table['model'] == model, numeric_cols].idxmax(axis=1).values[0]
            dataset.loc[i, 'Gear'] = max_category
    print("Missing Gear values filled.")

    df = dataset[['Engine_type', 'model']]
    chi_square_results = chi_square_test_all_pairs(df)
    print("Chi-square test for Engine type done.")
    
    pivot_table = dataset.pivot_table(index='model', columns='Engine_type', aggfunc='size', fill_value=0).reset_index()
    numeric_cols = pivot_table.columns[1:]
    pivot_table['Difference'] = pivot_table[numeric_cols].sum(axis=1) - pivot_table[numeric_cols].max(axis=1)
    print("Pivot table for Engine type created.")
    
    for i, row in dataset.iterrows():
        if pd.isnull(row['Engine_type']):
            try:
                model = row['model']
                max_category = pivot_table.loc[pivot_table['model'] == model, numeric_cols].idxmax(axis=1).values[0]
                dataset.loc[i, 'Engine_type'] = max_category
            except:
                dataset.loc[i, 'Engine_type'] = 'דיזל'
    print("Missing Engine type values filled.")
    
    dataset.loc[(dataset['capacity_Engine'] > 10000) & (dataset['model'] == 'אוקטביה'), 'capacity_Engine'] = 1200
    dataset.loc[(dataset['capacity_Engine'] > 10000) & (dataset['model'] == 'ג\'ולייטה'), 'capacity_Engine'] = 1400
    dataset.loc[(dataset['capacity_Engine'] > 10000) & (dataset['model'] == 'M1'), 'capacity_Engine'] = 3000
    print("Capacity engine outliers fixed.")
    
    group = dataset.groupby(['model'])['capacity_Engine'].agg(['median']).reset_index()
    
    for index, row in dataset.iterrows():
        if pd.isna(row['capacity_Engine']):
            model = row['model']
            median_value = group.loc[group['model'] == model, 'median'].values[0]
            dataset.at[index, 'capacity_Engine'] = median_value
    print("Missing capacity engine values filled.")

    dataset = dataset.loc[dataset['Year'] > 2003]
    print("Filtered dataset for years > 2003.")
    
    lst_km = []
    for i in range(len(dataset)):
        if dataset.iloc[i]['Km'] <= 60000:
            lst_km.append("<60000")
        elif 60000 < dataset.iloc[i]['Km'] <= 120000:
            lst_km.append("120000>x>60000")
        else:
            lst_km.append("120000<x")
    dataset['Km_group'] = lst_km
    print("Km groups created.")
    
    # שמירת עמודות נדרשות
    try:
        features = ['Km_group', 'model', 'Year', 'Gear', 'manufactor', 'Price']
        dataset = dataset[features]
    except KeyError:
        features = ['Km_group', 'model', 'Year', 'Gear', 'manufactor']
        dataset = dataset[features]
    print('Features selected.')
    
    # המרת עמודות קטגוריאליות ל-get dummies
        # המרת עמודות קטגוריאליות ל-get dummies
    dataset = pd.get_dummies(dataset)
    print("Categorical variables converted to dummies.")
    
    # בדיקה אם עמודת Price קיימת
    if 'Price' in dataset.columns:
        # שמירת עמודת Price בנפרד
        price = dataset['Price']
        dataset = dataset.drop(columns=['Price'])
        dataset['Price'] = price
        print("Price column moved to the end.")
    
    print("Data preparation finished.")
    return dataset







