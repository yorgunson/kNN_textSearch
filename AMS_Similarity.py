import pandas as pd
import TextSearch_MainClass as SimClass

# Start with creating a Text Similarity Instance
dbase_path='/Users/syor0001/Documents/Scrape/Text_scikit/TextSimilarity/dbase/AMS/'
ts=SimClass.textSIM(dbase_path)
analysis_column='Abstract'

#################
# DATABASE WORK #
#################
df=ts.read_df('JOURNAL_df')
print ('Raw Dataframe Size=',df.index.size)

# Filter/Clean the Database if Needed
mask = (df['Abstract'].str.len() > 30)
df = df.loc[mask]
print ('Filtered Dataframe Size=',df.index.size)

# Read target text and add to the database
shconv_abst=ts.read_text('shconv.txt')
prein_abst=ts.read_text('prein.txt')
bergemann2015_abst=ts.read_text('bergemann_2015.txt')
bergemann2016_abst=ts.read_text('bergemann_2016.txt')
bergemann2017_abst=ts.read_text('bergemann_2017.txt')

data_text=[['QJRM','1988','Tropical forecasting at ECMWF: The influence of physical parametrization on the mean structure of forecasts and analyses','M. Tiedtke','','',shconv_abst,''],
            ['REVG','2015','A review on regional convection-permitting climate modeling: Demonstrations, prospects, and challenges','Prein','','',prein_abst,''],
            ['JCLIM','2015','Global Detection and Analysis of Coastline-Associated Rainfall Using an Objective Pattern Recognition Technique','Bergemann et al.','','',bergemann2015_abst,''],
            ['GRL','2016','How important is tropospheric humidity for coastal rainfall in the tropics?','Bergemann and Jakob','','',bergemann2016_abst,''],
            ['JAMES','2017','Coastal Tropical Convection in a Stochastic Modeling Framework','Bergemann et al.','','',bergemann2017_abst,'']]

df=ts.add_textTOdf(df,data_text)
print ('Analysis Dataframe Size=',df.index.size)

# CREATE THE RAW DATA LIST FOR SIMILARITY CALCULATIONS
raw_data=df[analysis_column].tolist()

####################
# SIMILARITY WORK  #
####################

file_prefix='ams'       # The files to write/read. Depends on the project
n_gram_type='mix'       # Decides if the n-grams will be 'single' [(1,1),(2,2),(3,3)...] OR 'mix' [(1,1),(1,2),(1,3)...]
n_gram_max=4            # Max number of n in n-grams

# save_analyze  = Save Files First and Analyze
# save_only     = Save Files and DONT Analyze 
# prevfiles     = Work on Previously Saved Files 
work_type='prevfiles'

ind_list=[]
print ('Vectorization, TF-IDF Calculation')
print ('##################################')
for i in range(1,n_gram_max+1):
    
    # Target text index
    target_idx=0
    
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
        kywd='sea breeze'
        coeff=20
        tfidf_matrix=ts.scale_feature(tfidf_matrix,features,kywd,target_idx,coeff)
    
        # Show the common n-grams of the target text
        ts.show_common_ngrams(tfidf_matrix,features,target_idx,10)
        
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
        
    #######################
    #### PRINT RESULTS ####
    #######################
    print_columns=['Title','Link','Date']
    save=True
    ts.print_results(fin_list,df,analysis_column,print_columns,target_idx,kywd,file_prefix,n_gram_type,dbase_path,save)
    
