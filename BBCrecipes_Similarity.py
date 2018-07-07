import pandas as pd
import TextSearch_MainClass as SimClass
    
# Start with creating a Text Similarity Instance
dbase_path='/Users/syor0001/Documents/Scrape/Text_scikit/TextSimilarity/dbase/BBCrecipes/'
ts=SimClass.textSIM(dbase_path)
analysis_column='Method'

#################
# DATABASE WORK #
#################
df=ts.read_df('BBCrecipes_df')
print ('Raw Dataframe Size=',df.index.size)

# Filter/Clean the Database if Needed
mask = (df[analysis_column].str.len() > 30)
df = df.loc[mask]
print ('Filtered Dataframe Size=',df.index.size)

# Read target text and add to the database
Laksa=ts.read_text('laksa.txt')                       # Malaysian
CullenSkink=ts.read_text('CullenSkink.txt')           # Scottish 
Pmusakka=ts.read_text('patlican_musakka.txt')         # Turkish
Karniyarik=ts.read_text('Karniyarik.txt')             # Turkish
Barigoule=ts.read_text('Barigoule.txt')               # French
SpinachLeekPilau=ts.read_text('SpinachLeekPilau.txt') # African, but maybe Indian too

Laksa_ing=ts.read_text('laksa_ing.txt')                       
CullenSkink_ing=ts.read_text('CullenSkink_ing.txt')           
Pmusakka_ing=ts.read_text('patlican_musakka_ing.txt')
Karniyarik_ing=ts.read_text('Karniyarik_ing.txt') 
Barigoule_ing=ts.read_text('Barigoule_ing.txt')               
SpinachLeekPilau_ing=ts.read_text('SpinachLeekPilau_ing.txt') 

data_text=[['Manual','Malaysian','https://www.sbs.com.au/food/recipes/malaysian-laksa-cheats-laksa','Malaysian Laksa','15 mins','15 mins','Serves 6','None',Laksa_ing,Laksa],
           ['Manual','Scottish','https://www.sbs.com.au/food/recipes/cullen-skink','Cullen Skink','15 mins','15 mins','Serves 4','None',CullenSkink_ing,CullenSkink],
           ['Manual','Turkish','http://www.turkishyummies.com/eggplant-mousaka-patlican-musakka','Patlican Musakka','20 mins','20 mins','Serves 6','None',Pmusakka_ing,Pmusakka],
           ['Manual','French','https://www.saveur.com/article/recipes/barigoule-of-spring-vegetables','Barigoule','30 mins','30 mins','Serves 4','Vegetarian',Barigoule_ing,Barigoule],
           ['Manual','African','https://www.sbs.com.au/food/recipes/spinach-and-leek-pilau','Spinach and Leek Pilau','15 mins','55 mins','Serves 4','Vegetarian',SpinachLeekPilau_ing,SpinachLeekPilau],
           ['Manual','Turkish','https://www.sbs.com.au/food/recipes/stuffed-eggplant-lamb-or-beef-karniyarik','Karniyarik','15 mins','1 hour','Serves 6','None',Karniyarik_ing,Karniyarik]]

df=ts.add_textTOdf(df,data_text)
print ('Analysis Dataframe Size=',df.index.size)

# CREATE THE RAW DATA LIST FOR SIMILARITY CALCULATIONS
raw_data=df[analysis_column].tolist()

####################
# SIMILARITY WORK  #
####################

file_prefix='bbcrecipes'   # The files to write/read. Depends on the project
n_gram_type='mix'           # Decides if the n-grams will be 'single' [(1,1),(2,2),(3,3)...] OR 'mix' [(1,1),(1,2),(1,3)...]
n_gram_max=4                # Max number of n in n-grams

# save_analyze  = Save Files First and Analyze
# save_only     = Save Files and DONT Analyze 
# prevfiles     = Work on Previously Saved Files 
work_type='save_analyze'


ind_list=[]
print ('Vectorization, TF-IDF Calculation')
print ('##################################')
for i in range(1,n_gram_max+1):
    
    # Target text index
    target_idx=1
    
    # Arrange n-grams
    if n_gram_type=='single':k=i
    elif n_gram_type=='mix':k=1
        
    # Vectorize and Calculate Tf-idf. Extract features
    print ('------------------------------')
    print ('Working on '+str(k)+','+str(i))
    print ('------------------------------')
    
    if work_type=='save_only' or work_type=='save_analyze':
        
        vectors,tfidf_matrix,features=ts.vectorize_sw(raw_data,(k,i))
        
        # Save and Pickle
        ts.write_pickle(file_prefix+'_features'+str(k)+str(i),features)
        ts.write_pickle(file_prefix+'_tfidf'+str(k)+str(i),tfidf_matrix)
    
    elif work_type=='prevfiles':
        # Unpickle and Read
        tfidf_matrix=ts.read_pickle(file_prefix+'_tfidf'+str(k)+str(i)+'.gz')
        features=ts.read_pickle(file_prefix+'_features'+str(k)+str(i)+'.gz')
    
    # Continue Analysis 
    if work_type=='save_analyze' or work_type=='prevfiles':
        
        # Scale a keyword
        #kywd=None
        kywd='spinach'
        coeff=20
        tfidf_matrix=ts.scale_feature(tfidf_matrix,features,kywd,target_idx,coeff)
    
        # Show the common n-grams of the target text
        ts.show_common_ngrams(tfidf_matrix,features,target_idx,30)
        
        # Find neighbors and get the distances and indices of the neighbors
        distances,indices=ts.get_kneighbors(30,'minkowski',tfidf_matrix,target_idx)
        print ('Found Neighbors')
        ind_list.append(indices[0][1:])
        
# Merge and Present Results 
if len(ind_list)>0:
    
    # Merge results of different n-gram combinations (Comment in/out depending on the type merger)
    coeffs=[1,1,1,1]
    #coeffs=[1.25,1.75,2.5,2.5] # The coefficients corresponding to each n-gram

    fin_list=ts.results_merger(ind_list,coeffs)      # Merger
    #print (fin_list)
        
    #######################
    #### PRINT RESULTS ####
    #######################
    print_columns=['Name','Link','Cuisine']
    save=True
    ts.print_results(fin_list,df,analysis_column,print_columns,target_idx,kywd,file_prefix,n_gram_type,dbase_path,save)
