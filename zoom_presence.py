import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import pickle

class ZoomPresence:
    def __init__(self, file_path, meeting_start, model_path="model.pkl"):
        self.file_path = file_path
        self.model_path = model_path
        self.meeting_start = meeting_start
        self.date = None
        self.load_csv()
        self.load_word_vector()
        self.get_presence_date()
        
    def load_csv(self):
        self.presence = pd.read_csv(self.file_path)
        print(f"File {self.file_path} loaded successfuly!")
        
    def load_word_vector(self):
        """Loads tfidf word vector"""
        f = open(self.model_path, 'rb')
        self.tfidf, self.db_names, self.db_word_vec = pickle.load(f)
        f.close()
        print("Word vector loaded succesfuly!")
        
    def get_presence_date(self):
        self.date = self.presence["Join Time"][0][:10]
        m, d, y = self.date.split("/")
        self.output_name = f"processed-presence-{y}-{m}-{d}.xlsx"
        self.sheet_name = self.date.replace("/", "")
        
    def identify(self, zoom_name, return_similarity=False):
        """Identify the name and class of the participants related to the database"""
        zoom_name = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', " ", zoom_name)
        wv_nama = self.tfidf.transform([zoom_name])

        sim = cosine_similarity(wv_nama, self.db_word_vec)
        similarity_idx = sim.argsort()
        similarity_idx = similarity_idx.flatten()
        similarity_score = sim.sort()
        similarity_score = sim.flatten()

        if similarity_score[-1] < 0.5:
            if return_similarity:
                return "Unknown", similarity_score[-1]

            return "Unknown"
        
        if return_similarity:
            return self.db_names[similarity_idx[-1]], similarity_score[-1]
    
        return self.db_names[similarity_idx[-1]]
    
    def is_late(self, join_time):
        meeting_start = f"{join_time[:10]} {self.meeting_start}"
        meeting_start = datetime.strptime(meeting_start, "%m/%d/%Y %I:%M %p")
        
        if "AM" in join_time or "PM" in join_time:
            if len(join_time) >= 21:
                join_time = datetime.strptime(join_time, "%m/%d/%Y %I:%M:%S %p")
            else:
                join_time = datetime.strptime(join_time, "%m/%d/%Y %I:%M %p")
            
        else:
            if len(join_time) >= 19:
                join_time = datetime.strptime(join_time, "%m/%d/%Y %H:%M:%S")
            else:
                join_time = datetime.strptime(join_time, "%m/%d/%Y %H:%M")
            
        result = join_time - meeting_start
        result = result.total_seconds()/60
        return result >= 30
        
    def gen_identified_as(self):
        identified_as = []
        similarity = []
        full_name = []
        identified_class = []
        for i in self.presence["Name (Original Name)"]:
            a, b = self.identify(i, return_similarity=True)
            identified_as.append(a)
            similarity.append(b)
            
            if a == "Unknown":
                full_name.append(i)
                identified_class.append("Unknown")
            else:
                a, b = self.split_name_and_class(a)
                full_name.append(a)
                identified_class.append(b)
        
        self.presence.insert(1, "Identified As", identified_as)
        self.presence.insert(2, "Similarity", similarity)
        self.presence.insert(3, "Full Name", full_name)
        self.presence.insert(4, "Identified Class", identified_class)
        
        print("`Identified As`, `Similarity`, `Full Name`, and `Identifed Class` column added successfuly!")
    
    def split_name_and_class(self, name):
        a = re.search("\[", name)
        e = a.span()[0]-1
        return name[:e], name[e+2:-1]
        
    def gen_is_late(self):
        student_is_late = [self.is_late(join) for join in self.presence["Join Time"]]
        self.presence.insert(1, "Is Late", student_is_late)
        print("Is Late column added successfuly!")
        
    def gen_good_presence(self):
        self.gen_identified_as()
        self.gen_is_late()
        self.get_presence_date()
        known = self.presence.groupby(["Identified As"], as_index=False).agg({"Name (Original Name)":"max", "Similarity": "max", "Is Late":"min", "Full Name":"max", "Identified Class":"max", "Join Time": "min","Leave Time": "max", "Duration (Minutes)": "sum"})
        known.drop(known[known["Identified As"] == "Unknown"].index, inplace=True)
        unknown = self.presence[self.presence["Identified As"] == "Unknown"].groupby(["Identified As","Name (Original Name)"], as_index=False).agg({"Similarity": "max", "Is Late":"min", "Full Name":"max", "Identified Class":"max", "Join Time": "min","Leave Time": "max", "Duration (Minutes)": "sum"})
        self.good_presence = pd.concat([known, unknown], ignore_index=True)
        print("Good presence successfuly generated, free from duplicates and it's clean!")
    
    def save_as_excel(self, prefix_path, file_name):
        self.good_presence.index += 1
        self.good_presence.to_excel(f"{prefix_path}/{file_name}", self.sheet_name, index_label="Nomor", startrow=1)
        print(f"File {self.output_name} saved succesfuly!")