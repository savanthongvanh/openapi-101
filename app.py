from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. vectorise the products csv file using CSVLoader
loader = CSVLoader(file_path="products.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_responses = db.similarity_search(query, k=10)
    page_contents = [doc.page_content for doc in similar_responses]
    # print(page_contents)

    return page_contents

query = "what is the lightest backpack from northface?"

result = retrieve_info(query)
print(result)

# ['Product: Backpack\nWeight (in kg): 1.5\nBrand: Osprey', 
#  'Product: Camping Backpack Laptop Sleeve\nWeight (in kg): 0.3\nBrand: Osprey', 
#  'Product: Camping Backpack Laptop Sleeve\nWeight (in kg): 0.3\nBrand: Osprey']