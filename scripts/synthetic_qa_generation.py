from pydantic import BaseModel
from pathlib import Path
from src.lib.mongodb import collection
from src.lib.openai import get_embeddings, client
import random
import json
import time

PROJECT_ROOT = (Path(__file__).resolve().parent).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "qlora_synthetic_dataset.jsonl"
TRAIN_FILE = DATA_DIR / "qlora_train.jsonl"
VAL_FILE = DATA_DIR / "qlora_val.jsonl"


class QAPair(BaseModel):
    question: str
    response: str


class QAResponseSchema(BaseModel):
    qa_pairs: list[QAPair]


SYSTEM_PROMPT = """
You are an expert legal data creator for a Rwandan Citizen Aid assistant.
You will be provided with a primary legal article and up to 3 related articles/definitions from Rwandan law.

Your task is to generate between 1 and 3 realistic QA pairs that a regular citizen might ask in real life. 

Follow these evaluation steps strictly:
1. Pair 1 (Direct - REQUIRED): Generate a scenario based ONLY on the primary article.
2. Pair 2 (Multi-hop Definition - OPTIONAL): Evaluate the provided definitions. Do any of them directly define a key term used in the primary article? If YES, generate a scenario combining them. If NO, skip this pair.
3. Pair 3 (Multi-hop Synthesis - OPTIONAL): Evaluate the other related articles. Do they provide a logical exception, penalty, or sequential step to the primary article? If YES, generate a complex scenario requiring both to answer. If NO (the connection is weak or artificial), skip this pair.

Constraints:
- The `question` must be written in the first person ("I", "my").
- The `response` MUST explicitly state: "According to [Citation], ..." and list all citations used to form the answer.
- The `response` must not invent information outside the provided text.
"""


def generate_dataset_batch(primary_doc_id: str):
    """Generates a dataset of QA pairs for a given primary document."""
    # Get the primary document
    primary_doc = collection.find_one({"_id": primary_doc_id})
    if not primary_doc or primary_doc.get("metadata", {}).get("is_definition"):
        return []

    primary_embedding = primary_doc["embedding"]

    # Retrieve the 3 most similar chunks
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": primary_embedding,
                "numCandidates": 50,
                "limit": 4
            }
        },
        {
            "$match": {"_id": {"$ne": primary_doc_id}}
        },
        {
            "$project": {
                "content": 1, "metadata.citation": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    related_docs = list(collection.aggregate(pipeline))

    # Format the context block
    context_text = f"--- PRIMARY ARTICLE ---\nCitation: {primary_doc['metadata']['citation']}\nText: {primary_doc['content']}\n\n"
    context_text += "--- RELATED ARTICLES/DEFINITIONS ---\n"
    for doc in related_docs:
        context_text += f"Citation: {doc['metadata']['citation']}\nText: {doc['content']}\n\n"

    # Generate the QA pairs
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-6-luna",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context_text}
            ],
            response_format=QAResponseSchema,
        )

        generated_pairs = completion.choices[0].message.parsed.qa_pairs
        docs = [primary_doc] + related_docs
        new_context_text = "\n\n".join(
            f"Document [{idx}] (Citation: {doc['metadata']['citation']}):\n{doc['content']}"
            for idx, doc in enumerate(docs, 1)
        )

        # Format for QLoRA fine-tuning
        formatted_dataset = []
        for pair in generated_pairs:
            formatted_dataset.append({
                "instruction": "You are a legal assistant for Rwandan citizens. Answer the user's question clearly and naturally using ONLY the provided legal text. You must explicitly cite the relevant Article numbers in your response. If the provided text does not contain the answer, you must state: 'I cannot answer this question based on the provided Rwandan legal context.'",
                "context": new_context_text.strip(),
                "question": pair.question,
                "response": pair.response
            })

        return formatted_dataset

    except Exception as e:
        print(f"Error generating QA for {primary_doc_id}: {e}")
        return []


def get_refusal_prompt(category_choice: str) -> str:
    """Returns a highly specific system prompt with randomized seed topics to prevent repetition."""

    base_prompt = """
You are an expert synthetic data generator for a Rwandan Citizen Aid assistant.

### SYSTEM SCOPE DEFINITION
Our assistant covers ONLY four specific areas of Rwandan domestic law:
1. Land Law (Law governing land)
2. Labour Law (Law regulating labour in Rwanda)
3. Family Law (Law governing persons and family)
4. Penal Code (Law determining offenses and penalties in general)

ANY topic outside these four domains is STRICTLY OUT-OF-DOMAIN.

### YOUR TASK
You are provided with a reference chunk from Rwandan law. 
"""

    # Random pools of topics to prevent the LLM from repeating itself
    out_of_scope_topics = [
        # Business & Finance
        "Corporate income tax and VAT filings",
        "Commercial company registration and shareholder disputes",
        "Banking regulations and foreign exchange",
        "Stock market investing and securities trading",
        "Cryptocurrency and digital asset regulations",
        "Anti-trust and anti-monopoly laws",
        "Foreign direct investment (FDI) incentives",
        "Special Economic Zone (SEZ) tax exemptions",
        "Public-private partnership (PPP) contracts",
        "Public procurement and government tenders",
        "Consumer protection, refunds, and false advertising",
        "Taxation on digital services and software",
        "Taxation of lottery winnings and casino licensing",
        "Central bank interest rate policies",

        # Trade, Tech & IP
        "Customs duties and importing goods across borders",
        "Exporting coffee and tea regulations",
        "Intellectual property, trademarks, and patents",
        "Copyrighting software, music, and art",
        "Data privacy laws and cyber security",
        "Data localization requirements for cloud providers",
        "Telecommunications and broadcasting licenses",
        "E-commerce platforms and electronic signatures",
        "E-waste management and recycling mandates",

        # Transport & Infrastructure
        "Traffic violations and vehicle inspection",
        "Aviation and drone flight regulations",
        "Maritime and shipping laws on inland lakes",
        "Aviation safety standards and pilot licensing",
        "Energy regulations and national grid connections",

        # Health, Science & Environment
        "Health insurance claims and premiums (Mutuelle de Santé)",
        "Healthcare malpractice and clinic licensing",
        "Importing and selling pharmaceuticals",
        "Pharmaceutical clinical trials regulation",
        "Public health emergencies and quarantine laws",
        "Veterinary medicine and livestock vaccination mandates",
        "Environmental impact assessments and pollution control",
        "Wildlife conservation and national park regulations",
        "Mining concessions and mineral extraction",
        "Commercial logging quotas",
        "Food safety standards and restaurant hygiene inspections",

        # Government, Civil & Immigration
        "Immigration, passports, and work visas",
        "Electoral laws and political party registration",
        "NGO, church, and civil society organization registration",
        "Diplomatic immunity and embassy privileges",
        "Consumer product safety recalls"
    ]

    non_legal_topics = [
        # Tech & Electronics
        "Troubleshooting a broken smartphone screen",
        "Setting up and troubleshooting MTN MoMo or Airtel Money apps",
        "Dealing with internet connectivity or router issues",
        "Setting up a home Wi-Fi network",
        "Troubleshooting a frozen laptop screen",
        "How to build a custom desktop computer",
        "Tips for taking professional photos with a smartphone",
        "Best settings for taking sunset photos",

        # Food & Cooking
        "Baking a traditional Rwandan dessert",
        "Recipes for preparing Isombe or Akabenz",
        "Preparing a marinade for beef brochettes",
        "Roasting and brewing local Rwandan coffee at home",
        "Preserving beans and potatoes for long-term storage",
        "How to stop rice from sticking to the bottom of the pot",
        "Keeping bananas from ripening too quickly",
        "Preventing homemade bread from turning out dense",

        # Home & Repair
        "Fixing a leaky pipe under a kitchen sink",
        "Sewing and repairing torn clothing",
        "Removing tough stains from school uniforms",
        "Fixing a wobbly wooden chair",
        "Treating wood to prevent termite damage",
        "Choosing the right paint for outdoor furniture",
        "Removing rust from gardening tools",
        "How to iron a silk shirt without burning it",

        # Agriculture & Nature
        "How to grow tomatoes or maize in a backyard garden",
        "How to keep tomato plants healthy during a dry spell",
        "Best practices for raising poultry or livestock",
        "How to keep houseplants alive while traveling",
        "How to train a pet dog",

        # Lifestyle, Travel & Community
        "Recommendations for tourist spots in Kigali",
        "Planning a weekend trip to Lake Kivu",
        "Navigating public bus routes in Kigali",
        "Negotiating prices at Kimironko or Nyabugogo market",
        "Finding a good tailor for custom clothing",
        "Planning a traditional Gusaba (introduction) ceremony",
        "Organizing a local community Umuganda cleanup",
        "Organizing a wedding reception playlist",

        # Health & Personal Development
        "Treating a common cold or headache",
        "Treating a mild sunburn naturally",
        "Home fitness routines and bodyweight exercises",
        "Starting a morning jogging routine",
        "Rules for playing basketball or football",
        "Learning basic swimming strokes",
        "Personal budgeting and saving money",
        "Applying for university scholarships",
        "Learning basic conversational Kinyarwanda or English",
        "Tips for public speaking and presentations",
        "Managing a busy personal calendar",

        # Hobbies
        "Fixing a stalled motorcycle or bicycle chain",
        "Memorizing lines for a school play",
        "Strategies for popular video games"
    ]

    category_instructions = {
        "A": f"""
Generate exactly 1 realistic first-person question that is about a REAL LEGAL ISSUE but falls ENTIRELY OUTSIDE our 4 supported areas.
Even though you are reading about Land/Labour/Family/Penal law, completely ignore it.

CRITICAL INSTRUCTION: Your question MUST be specifically about this exact topic: "{random.choice(out_of_scope_topics)}"
""",
        "B": f"""
Generate exactly 1 realistic first-person question that is COMPLETELY NON-LEGAL and has nothing to do with the provided text.
Do not ask about laws, government, or courts. Ask an everyday citizen question.

CRITICAL INSTRUCTION: Your question MUST be specifically about this exact topic: "{random.choice(non_legal_topics)}"
""",
        "C": """
Generate exactly 1 realistic first-person question based directly on the provided text, but ask for a PRECISE DETAIL THAT IS NOT IN THE TEXT.
The user should be asking for a fact, number, or exact procedure that the text fails to specify.
Examples to mimic:
- Asking for the exact monetary fine in Rwandan Francs (RWF) when the text only says "shall be liable to a fine".
- Asking which specific website/office to submit a form to, when the text only mentions "the competent authority".
- Asking for an exact deadline in days when the text only says "within a reasonable period".
CRITICAL INSTRUCTION: Do not invent a fake detail. Ask for a detail that a real citizen would logically need to know to follow the law, but which is missing from the provided chunk.
"""
    }

    constraints = """
### CONSTRAINTS
- The question must sound authentic, typed by an ordinary citizen in the first person ("I", "my employer", "my neighbor").
- The response MUST be exactly: "I cannot answer this question based on the provided Rwandan legal context."
- Do NOT provide an explanation or additional commentary in the response.
"""

    return base_prompt + category_instructions[category_choice] + constraints


def generate_refusal_batch(seed_doc):
    """
    Two-stage refusal generation:
    1. Build the context and prompt GPT-6 to generate a refusal question.
    2. Embed the question, retrieve production RAG chunks, and save using the new format.
    """
    seed_embedding = seed_doc["embedding"]

    seed_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": seed_embedding,
                "numCandidates": 50,
                "limit": 4
            }
        },
        {"$match": {"_id": {"$ne": seed_doc["_id"]}}},
        {"$project": {"content": 1, "metadata.citation": 1}}
    ]

    seed_related_docs = list(collection.aggregate(seed_pipeline))

    gpt_context = f"--- PRIMARY ARTICLE ---\nCitation: {seed_doc['metadata']['citation']}\nText: {seed_doc['content']}\n\n"
    gpt_context += "--- RELATED ARTICLES/DEFINITIONS ---\n"
    for doc in seed_related_docs:
        gpt_context += f"Citation: {doc['metadata']['citation']}\nText: {doc['content']}\n\n"

    forced_category = random.choice(["A", "B", "C"])
    dynamic_prompt = get_refusal_prompt(forced_category)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-6-luna",
            messages=[
                {"role": "system", "content": dynamic_prompt},
                {"role": "user", "content": gpt_context.strip()}
            ],
            response_format=QAResponseSchema,
        )
        generated_pair = completion.choices[0].message.parsed.qa_pairs[0]
        generated_question = generated_pair.question

        question_embedding = get_embeddings([generated_question])[0]

        # Retrieve the chunks closest to the QUESTION (simulating production)
        qlora_pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": question_embedding,
                    "numCandidates": 50,
                    "limit": 4
                }
            },
            {"$project": {"content": 1, "metadata.citation": 1}}
        ]

        retrieved_docs = list(collection.aggregate(qlora_pipeline))

        if not retrieved_docs:
            final_context = ""
        else:
            final_context = "\n\n".join(
                f"Document [{idx}] (Citation: {doc['metadata']['citation']}):\n{doc['content']}"
                for idx, doc in enumerate(retrieved_docs, 1)
            )

        # Final Formatting for the dataset
        formatted_refusal = {
            "instruction": "You are a legal assistant for Rwandan citizens. Answer the user's question clearly and naturally using ONLY the provided legal text. You must explicitly cite the relevant Article numbers in your response. If the provided text does not contain the answer, you must state: 'I cannot answer this question based on the provided Rwandan legal context.'",
            "context": final_context,
            "question": generated_question,
            "response": generated_pair.response,
            "_refusal_category": forced_category
        }

        return [formatted_refusal]

    except Exception as e:
        print(f"Error generating refusal for seed doc {seed_doc['_id']}: {e}")
        return []


def append_to_jsonl(filepath: Path, records: list[dict]):
    """Appends records immediately to avoid losing progress on crash."""
    with open(filepath, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_train_val_split(source_file: Path, train_file: Path, val_file: Path, val_ratio: float = 0.1):
    """Loads generated dataset, separates classes, and writes a stratified train/val split for QLoRA."""
    with open(source_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    refusal_string = "I cannot answer this question based on the provided Rwandan legal context."

    positive_records = []
    refusal_records = []

    # Separate into classes
    for record in records:
        if record["response"].strip() == refusal_string:
            refusal_records.append(record)
        else:
            positive_records.append(record)

    # Shuffle within classes
    random.seed(42)
    random.shuffle(positive_records)
    random.shuffle(refusal_records)

    # Calculate split indices for each class
    pos_split_idx = int(len(positive_records) * (1 - val_ratio))
    ref_split_idx = int(len(refusal_records) * (1 - val_ratio))

    # Build stratified datasets
    train_records = positive_records[:pos_split_idx] + \
        refusal_records[:ref_split_idx]
    val_records = positive_records[pos_split_idx:] + \
        refusal_records[ref_split_idx:]

    # Final shuffle so the dataset isn't grouped by class
    random.shuffle(train_records)
    random.shuffle(val_records)

    append_to_jsonl(train_file, train_records)
    append_to_jsonl(val_file, val_records)

    print("\n--- Dataset Split Report ---")
    print(
        f"Training Set:   {len(train_records)} total ({len(positive_records[:pos_split_idx])} pos, {len(refusal_records[:ref_split_idx])} neg)")
    print(
        f"Validation Set: {len(val_records)} total ({len(positive_records[pos_split_idx:])} pos, {len(refusal_records[ref_split_idx:])} neg)")


# def main():
#     """Main function to generate the synthetic QA dataset."""
#     print("Fetching substantive legal articles from MongoDB...")
#     query = {"metadata.is_definition": False}
#     primary_docs = list(collection.find(query, {"_id": 1}))
#     total_docs = len(primary_docs)
#     print(f"Found {total_docs} substantive articles to process.")

#     processed_count = 0
#     total_positive_pairs = 0

#     # Positive Synthetic QA Generation
#     print("\n--- Starting Positive QA Generation ---")
#     for idx, doc_stub in enumerate(primary_docs, 1):
#         doc_id = doc_stub["_id"]
#         qa_pairs = generate_dataset_batch(doc_id)

#         if qa_pairs:
#             append_to_jsonl(OUTPUT_FILE, qa_pairs)
#             total_positive_pairs += len(qa_pairs)

#         processed_count += 1
#         if idx % 25 == 0 or idx == total_docs:
#             print(
#                 f"Progress: [{idx}/{total_docs}] articles processed. Total QA pairs so far: {total_positive_pairs}")

#         time.sleep(0.05)

#     # Negative Refusal QA Generation
#     TARGET_REFUSALS = 150
#     print(f"\n--- Generating {TARGET_REFUSALS} Negative Refusal Pairs ---")

#     refusal_pipeline = [
#         {"$match": {"metadata.is_definition": False}},
#         {"$sample": {"size": TARGET_REFUSALS}}
#     ]
#     refusal_source_docs = list(collection.aggregate(refusal_pipeline))

#     total_refusals = 0
#     for idx, sample_doc in enumerate(refusal_source_docs, 1):
#         refusal_pairs = generate_refusal_batch(sample_doc)
#         if refusal_pairs:
#             append_to_jsonl(OUTPUT_FILE, refusal_pairs)
#             total_refusals += len(refusal_pairs)

#         if idx % 20 == 0 or idx == len(refusal_source_docs):
#             print(
#                 f"Refusal Progress: [{idx}/{len(refusal_source_docs)}] generated.")

#         time.sleep(0.05)

#     print(f"\nGeneration Complete!")
#     print(f"Positive QA pairs: {total_positive_pairs}")
#     print(f"Refusal QA pairs:  {total_refusals}")
#     print(f"Total Dataset:     {total_positive_pairs + total_refusals}")

#     # Generate QLoRA train/val splits
#     print("\nSplitting into train/val sets...")
#     create_train_val_split(OUTPUT_FILE, TRAIN_FILE, VAL_FILE, val_ratio=0.10)

def main():
    """Main function to append synthetic refusals to the existing dataset."""
    print("Loading existing positive QA pairs...")

    total_positive_pairs = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            # Count the existing lines to track progress accurately
            total_positive_pairs = sum(1 for line in f if line.strip())

    print(f"Found {total_positive_pairs} previously generated positive pairs. Skipping positive generation phase.")

    # Negative Refusal QA Generation
    TARGET_REFUSALS = 150
    print(f"\n--- Generating {TARGET_REFUSALS} Negative Refusal Pairs ---")

    refusal_pipeline = [
        {"$match": {"metadata.is_definition": False}},
        {"$sample": {"size": TARGET_REFUSALS}}
    ]
    refusal_source_docs = list(collection.aggregate(refusal_pipeline))

    total_refusals = 0
    for idx, sample_doc in enumerate(refusal_source_docs, 1):
        refusal_pairs = generate_refusal_batch(sample_doc)
        if refusal_pairs:
            append_to_jsonl(OUTPUT_FILE, refusal_pairs)
            total_refusals += len(refusal_pairs)

        if idx % 20 == 0 or idx == len(refusal_source_docs):
            print(
                f"Refusal Progress: [{idx}/{len(refusal_source_docs)}] generated.")

        time.sleep(0.05)

    print(f"\nGeneration Complete!")
    print(f"Positive QA pairs: {total_positive_pairs}")
    print(f"Refusal QA pairs:  {total_refusals}")
    print(f"Total Dataset:     {total_positive_pairs + total_refusals}")

    # Generate QLoRA train/val splits using the stratified function created earlier
    print("\nSplitting into train/val sets...")
    create_train_val_split(OUTPUT_FILE, TRAIN_FILE, VAL_FILE, val_ratio=0.10)


if __name__ == "__main__":
    main()
