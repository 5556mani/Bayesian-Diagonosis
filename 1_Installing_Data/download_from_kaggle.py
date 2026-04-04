import kagglehub
import os
import shutil


# 1. storing the location of kaggle dataset
dataset_handle="behzadhassan/sympscan-symptomps-to-disease"

def download_and_save():
    # 2. first we have to download it to deafult kagglehub cache 
    #    there is some setting that it only download to cache only by default,
    #    but i want to save it to my code folder
    print(f"Download process is started :{dataset_handle}")
    cache_path =kagglehub.dataset_download(dataset_handle)

    # I want to save the data in the same folder of where this downloading script will be
    script_dir =os.path.dirname(os.path.abspath(__file__))
    target_dir =os.path.join(script_dir,"data")

    # As we are saving data in taget_dir , but if there is not any "data" file there then we have to create it 
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Data directory is created under {target_dir}")


    # as by deafult our data is saved inside the cache we have to move it to target_dir
    print("Moving data from cache to project folder...")
    for filename in os.listdir(cache_path):
        source_file      =os.path.join(cache_path, filename)
        destination_file =os.path.join(target_dir,filename)
        # just checking whether it is a file or a folder

        if os.path.isfile(source_file):
            shutil.copy2(source_file,destination_file)
        elif os.path.isdir(source_file):
            if os.path.exists(destination_file):
                shutil.rmtree(destination_file)
            shutil.copy(source_file,destination_file)

    print(f"Download complete dataset is not at: {target_dir}")

# to make sure that it only run cant be called
if __name__=="__main__":
    download_and_save()
