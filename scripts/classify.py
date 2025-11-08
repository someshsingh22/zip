import pandas as pd

df = pd.read_csv("data/reddit/selected_users_v3_images_filtered.csv").dropna(
    subset=["title"]
)

flairs = [
    "Artwork",
    "Artwork ",
    "Showcase",
    "approved",
    "Arts/Crafts",
    "AI Showcase - Midjourney",
    "Midjourney :a2:",
    "Bing Image Creator :a2:",
    "Politics",
    "R1: Screenshot",
    "rule 3",
    "Stable Diffusion :a2:",
    "Discussion",
    "Other: Please edit :a2:",
    "V4 Showcase",
    "DALL E 3 :a2:",
    "Halloween",
    "Rule 1 - Title format",
    "Question",
    "Jokes/Meme",
    "R4: Title Guidelines",
    "R1: Text/emojis/scribbles",
    "Album",
    "R4: Inappropriate Title",
    "US Politics",
    "Starryai :a2:",
    "Rule 1",
    "Artwork CC",
    "Removed: R1",
    "rm: title guidelines",
    "Great Critique in Comments",
    "Video Art",
    "Rule 7",
    "R5: Title Rules",
    "Wombo :a2:",
    "R1: Text/Graphics",
    "Rule 5",
    "R1: No screenshots or pics where the only focus is a screen.",
    "R1: Screen",
    "Nightcafe :a2:",
    "Rule 7 - Fanart",
    "R1: Screen/Text",
    "Discussion :speaking2:",
    "Wonder :a2:",
    "Question :question:",
    "Rule 4 - Sketch/doodle/progress",
    "R1: Text/Comic",
    "R1: Text/Comic/Infographic",
    "ChatGPT :a2:",
    "DALL E 2 :a2:",
    "News Article",
    "Question - Midjourney AI",
    "Article",
    "AI Video + Midjourney",
    "Prompt-Sharing",
    "FLUX :a2:",
    "In The World",
    "rm: text/digital",
    "Jokes/Meme - Midjourney AI",
    "Midjourney",
    "Leonardo.ai :a2:",
    "Paintover/Edited",
    "Discussion - Midjourney AI",
    "Protest",
    "Resources/Tips",
    "Rule 12",
    "R10: No FCoO/Flooding",
    "Eclipse",
    "rm: screenshot",
    "picture of text",
    "r5: title guidelines",
    "Rule 9",
    "R1: Text",
    "Video :video:",
    "R5: title guidelines",
    "Election 2016",
    "Rule 4",
    "Rule 6",
    "Other",
    "R6: Indirect Link",
    "Not John Oliver – Removed",
    "Rule 5 - Objects in frame",
    "Nightcafe",
    "Rule 3",
    "No Animated Images",
    "In The World - Midjourney AI",
    "R5: Bad Host",
    "Picture of text",
    "Video",
    "R5: Inappropriate Title",
    "💩Shitpost💩",
    "Backstory",
    "News Article :newspaper:",
    "DeepAI :a2:",
    "backstory",
    "Paintover/Edited - Midjourney AI",
    "R1: screenshot",
    "Removed: R6",
    "R2: text/digital",
    "r1: screenshot/ai",
]

phrases = [
    "rule",
    "removed",
    "video",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "r7",
    "discussion",
    "question",
    "politics",
    "article",
    "meme",
    "protest",
    "rm:",
    "backstory",
    "picture",
    "in the world",
]
check = lambda x: any(phrase in x.lower() for phrase in phrases) or x in [
    "Other",
    "Resources/Tips",
    "Election 2016",
    "No Animated Images",
    "💩Shitpost💩",
]

art = df[(df["subreddit"] == "Art") | df["subreddit"] == "ArtPorn"]
art["title"] = art["title"].apply(lambda x: x.split(",")[0])

images = df[~df["subreddit"].isin(["Art", "ArtPorn"])]
images = images[
    images["link_flair_text"].isna()
    | (~images["link_flair_text"].fillna("").apply(check))
]
images = images[
    ~images["title"].apply(
        lambda x: x.endswith("?")
        or any(
            w in x.lower()
            for w in [
                "why",
                "how",
                "what",
                "should",
                "can",
                "does",
                "anyone",
                "help",
                "think",
                " i ",
                "i'm",
                " my ",
                " our ",
            ]
        )
    )
]
images = images[images["title"].apply(lambda x: len(x.split()) > 2)]
images = images[
    ~images["title"].apply(
        lambda x: x.lower().startswith("i'm")
        or x.lower().startswith("my")
        or x.lower().startswith("our")
        or x.lower().startswith("i ")
    )
]

df = pd.concat([images, art])
df.to_csv("data/reddit/selected_users_v3_images_filtered_classified.csv", index=False)
