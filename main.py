import TOF_SIMS as ts




if __name__ == "__main__":
    print("Loading a file")
    lead_bulk_file_path = "E:/Filename_2019.07.09-14h52m38s.h5"
    aluminum_thin_film_path = "E:/Filename_2019.04.11-09h51m35s.h5"
    #to open a file, simply pass the filepath to ts.TOF_SIM class
    #open the lead bulk TOF_SIM dataset

    #lead_bulk = ts.TOF_SIMS(lead_bulk_file_path)

    aluminum_thin_film = ts.TOF_SIMS(aluminum_thin_film_path)
    print("Done")
    #aluminum_thin_film.plot_FIBImages()
