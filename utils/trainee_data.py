
import pandas as pd


# Training data
base_data = {
    'text': [
        "I feel amazing today",
        "I'm so tired and down",
        "Things are confusing but I'm hopeful",
        "Excited for tomorrow!",
        "I hate everything",
        "Feeling neutral, not much happened"
    ],
    'label': ["Positive", "Negative", "Positive", "Positive", "Negative", "Neutral"]
}

extra_data = {
    'text': [
        "I love rainy days — they calm my mind.",
        "I feel anxious when people judge me.",
        "Grateful for the peace, though I miss my friends.",
        "They make me feel worthless sometimes.",
        "Joyful moments fade too quickly.",
        "Nothing exciting happened today.",
        "I'm proud of my progress.",
        "I pretend I'm fine but I'm not.",
        "Even in silence, I find comfort.",
        "Feeling stuck, unsure of everything.",
        "Rain worries me when I think of farmers.",
        "I have hope — it’s fragile, but still alive.",
        "I feel numb and detached",
        "They envy my success and treat me differently",
        "I worry about harvests when it rains",
        "I don't know how to feel lately"
    ],
    'label': [
        "Positive", "Negative", "Neutral", "Negative", "Neutral", "Neutral",
        "Positive", "Negative", "Positive", "Negative", "Neutral", "Positive",
        "Negative", "Positive", "Negative", "Negative"
    ]
}

df = pd.concat([pd.DataFrame(base_data), pd.DataFrame(extra_data)], ignore_index=True)

emotion_data = {
    'text': [
        "They cheered when I failed — betrayal has a sharp echo.",
        "I pretend I’m happy, but I feel forgotten in every room.",
        "No one asked if I’m okay, and I stopped expecting it.",
        "The silence after their goodbye is louder than their presence ever was.",
        "I hurt someone I love and now I carry it like a scar.",
        "I laugh so no one sees I’m breaking inside.",
        "I feel invisible even when I’m surrounded by people.",
        "I wish I hadn’t said those words — I regret every syllable.",
        "They envy my peace but don’t see the storm it hides.",
        "Sometimes I wonder if anyone would miss me if I disappeared.",
        "Everything is fine. Except I feel like I’m drowning quietly.",
        "I should’ve stood up for myself. It eats me inside now.",
        "They chose their convenience over our connection — that shattered me.",
        "I smiled today — not because I felt joy, but because it felt required.",
        "I’m strong because I have no choice. Not because I’m okay.",
        "I want to forgive myself, but guilt keeps rewinding the memory.",
        "Hope flickers — fragile, but still breathing somewhere inside.",
        "Why do they always get celebrated while I shrink in shadows?",
        "I feel nothing. Just blank space and delayed reactions.",
        "My body moves, my mouth talks — but I’m emotionally offline."
    ],
    'label': [
        "Betrayal", "Isolation", "Isolation", "Betrayal", "Guilt", "Numbness", "Isolation", "Regret",
        "Jealousy", "Isolation", "Numbness", "Regret", "Betrayal", "Numbness", "Numbness", "Guilt",
        "Hope", "Jealousy", "Numbness", "Numbness"
    ]
}

df_emotion=pd.DataFrame(emotion_data)
df=pd.concat([df,df_emotion],ignore_index=True)

emotion_data_2 = {
    'text': [
        "They asked if I was okay just to ease their conscience — not mine.",
        "Even when I speak, it feels like I’m not heard at all.",
        "I miss the old version of myself before all this heaviness settled in.",
        "She promised she'd stay — then ghosted me like I never mattered.",
        "Every compliment I hear feels like pity dressed up as kindness.",
        "I wear a smile so they stop asking what’s wrong.",
        "Their apology came too late — after I’d already shattered.",
        "I scroll for hours to escape my own thoughts.",
        "I feel surrounded by people, yet completely detached.",
        "I’m proud of myself for surviving a day I didn’t want to face.",
        "They laughed off my pain — like I’m the punchline of my own breakdown.",
        "I wish someone would care without asking me to perform strength first.",
        "I miss them more than I admit, but I act like I'm fine.",
        "They love the version of me that hides my feelings.",
        "Hope is all I have — it’s small, but it’s still mine.",
        "I feel like a ghost in rooms I once called home.",
        "Regret doesn’t knock — it barges in at 2 a.m.",
        "I was never part of their inner circle, just the outer convenience.",
        "Guilt eats quietly — it doesn't shout, but it stays.",
        "I’m learning to breathe again, even if it hurts at first.",
        "They envy my resilience, but never ask how it broke me.",
        "I act like nothing bothers me, even though it does — every time.",
        "Regret has a memory sharper than forgiveness.",
        "I’m not lazy — I’m just emotionally exhausted.",
        "Jealousy makes me hate things I used to love.",
        "I feel hope when I hear my own laughter return.",
        "They stole my softness and called it weakness.",
        "I forgive them — but I don't trust them anymore.",
        "I mourn people who are still alive, just emotionally distant.",
        "I’m not okay — I just know how to fake it better now."
    ],
    'label': [
        "Betrayal", "Isolation", "Regret", "Betrayal", "Guilt", "Numbness", "Betrayal", "Numbness", "Isolation", "Hope",
        "Betrayal", "Isolation", "Regret", "Numbness", "Hope", "Isolation", "Regret", "Isolation", "Guilt", "Hope",
        "Jealousy", "Numbness", "Regret", "Numbness", "Jealousy", "Hope", "Betrayal", "Regret", "Isolation", "Numbness"
    ]
}


df_emotion_2=pd.DataFrame(emotion_data_2)
df=pd.concat([df,df_emotion_2],ignore_index=True)

# Emotionally rich Joy entries to boost classifier sensitivity
emotion_joy_data = {
    'text': [
        "She hugged me so tightly it felt like the world paused — just warmth and love.",
        "We laughed until our stomachs hurt — I haven’t felt this light in ages.",
        "Today felt like a sunrise — calm, radiant, and full of promise.",
        "I danced barefoot in the rain — it was messy, magical, and I loved every second.",
        "His unexpected call made me smile for hours — it felt like connection again.",
        "We stayed up talking about dreams, and I felt truly heard.",
        "My cheeks hurt from smiling — this joy feels electric.",
        "I cooked my favorite meal while singing — my heart felt full.",
        "The way they looked at me made me forget everything else.",
        "Holding hands while watching the stars — it felt like the universe winked at me.",
        "I got the job! And it’s exactly what I hoped for. Still buzzing.",
        "My little brother said ‘I love you’ without prompting — I melted inside.",
        "I felt seen today. Not just noticed — truly understood.",
        "We clinked coffee mugs at 2 a.m. and giggled over nothing. Bliss.",
        "I sang in front of people for the first time. And I wasn’t scared — I felt alive.",
        "That moment when they pulled me into a spontaneous dance — unforgettable.",
        "They remembered my favorite dessert. I didn’t even ask.",
        "He kissed my forehead and I knew everything was okay.",
        "She held my hand through the bad news, and it made me feel safe.",
        "I got a random compliment from a stranger today — still smiling."
        "We twirled under fireworks, forgetting everything but each other.",
        "I danced while cooking and spilled everything — still laughed.",
        "We built a blanket fort and talked like kids again.",
        "She surprised me with coffee and my favorite playlist.",
        "I sang in the shower so loud the neighbors clapped.",
        "I felt proud — not just happy, proud of who I’m becoming.",
        "We shared secrets over ice cream — pure sweetness.",
        "My dog wagged at me like I was his whole world.",
        "I made someone laugh so hard they cried — best feeling ever.",
        "We joked all night until the sunrise joined us."   
    ],
    'label': ["Joy"] * 29
}

# Convert to DataFrame
df_joy = pd.DataFrame(emotion_joy_data)

# ✅ Concatenate with your main DataFrame
df = pd.concat([df, df_joy], ignore_index=True)

# Emotionally layered journal entries (edge cases)
emotion_edge_data = {
    'text': [
        "Sure, I’m totally fine — just cried in the bathroom, but hey, that’s adulthood.",
        "They said they missed me, but their silence screams louder than their words.",
        "Smiled all day — mostly so no one would ask what’s wrong.",
        "They laughed, and I pretended it didn’t cut me in half.",
        "I posted something positive today — no one noticed, but I smiled anyway.",
        "I joked about my breakdown so well they thought I was hilarious.",
        "Sometimes I miss people who never made space for me.",
        "He hugged me, and I almost forgot how broken I felt.",
        "I’m strong — because collapsing isn’t convenient right now.",
        "They compliment me when I’m quiet. I think silence makes me easier to love.",
        "I’m the kind of tired sleep doesn’t fix.",
        "They listened, but only to reply — never to understand.",
        "I smiled while reading their message. Then stared at the wall for an hour.",
        "I wish ‘I’m okay’ meant something when I say it.",
        "Being the ‘strong one’ is starting to feel like being invisible.",
        "I laughed out loud… then wiped away tears no one saw.",
        "They remembered my birthday — with a forwarded text.",
        "I said ‘thank you’ when they left. Not because I meant it.",
        "I’m not hiding pain — I just don’t know how to show it anymore.",
        "I forgave them. But I still rehearse the things I should’ve said."
    ],
    'label': [
        "Sarcasm", "Isolation", "Numbness", "Betrayal", "Disappointment",
        "Sarcasm", "Regret", "Bittersweet", "Numbness", "Sadness",
        "Exhaustion", "Isolation", "Numbness", "Suppression", "Guilt",
        "Sarcasm", "Disappointment", "Sarcasm", "Numbness", "Regret"
    ]
}

# Convert to DataFrame
df_edge = pd.DataFrame(emotion_edge_data)

# ✅ Concatenate with existing DataFrame
df = pd.concat([df, df_edge], ignore_index=True)

# Bittersweet emotional entries to deepen classifier nuance
emotion_bittersweet_data = {
    'text': [
        "She smiled at me through tears — it was beautiful and it broke me.",
        "We danced like everything was okay, even though nothing was.",
        "The sunset was stunning today — I watched it alone.",
        "Their hug felt warm, but it also reminded me how much I've missed them.",
        "I laughed so hard I forgot my pain for a moment — just a moment.",
        "They were kind today. Maybe they’re trying. Maybe I’m still hurt.",
        "His voice still makes my heart race — even if it's not for me anymore.",
        "I finally let myself feel happy — and guilt whispered that I shouldn’t.",
        "We had coffee together like old times. I smiled through the ache.",
        "That song reminded me of better days — I let it play twice.",
        "We talked all night. It felt like closure and a quiet goodbye.",
        "Her text made me smile and realize how much I missed her.",
        "I saw them laughing without me. They looked happy. I felt hollow.",
        "They remembered something small I love — I cried after they left.",
        "I kissed him knowing it might be the last time. I still did it.",
        "I felt peace today — fragile, flickering, but mine.",
        "We were close again, like always. It still hurt when they walked away.",
        "I said 'thank you for being here' and silently added 'for now.'",
        "They gave me hope and distance in the same sentence.",
        "I stared at the ceiling after laughing with them — unsure if I was happy or hurting."
    ],
    'label': ["Bittersweet"] * 20
}

# Convert to DataFrame
df_bittersweet = pd.DataFrame(emotion_bittersweet_data)

# ✅ Concatenate with your main DataFrame
df = pd.concat([df, df_bittersweet], ignore_index=True)

suppression_data = {
    'text': [
        "I say I’m fine so many times I almost believe it.",
        "My smile hides more than it shows.",
        "I bottle everything up — silence feels safer.",
        "I changed the subject so they wouldn’t see me unravel.",
        "I tell people I'm okay, even when I'm breaking quietly.",
        "Every time they ask, I smile instead of answer.",
        "I learned how to shrink my truth into a laugh.",
        "I feel everything — and say nothing.",
        "If I cry, they’ll think I’m weak. So I don’t.",
        "Even when I need help, I act like I don’t.",
        "I put on strength like makeup — just enough to look normal.",
        "They don’t notice my pain because I always look composed.",
        "I cover my stress in jokes so they stop asking.",
        "If I open up, I won’t stop. So I stay closed.",
        "I flinch internally and joke externally.",
        "My silence isn't peace — it's restraint.",
        "I carry everything inside and call it coping.",
        "I hide my true reactions to keep the room calm.",
        "I nod along even when I feel misunderstood.",
        "I hold back tears like it's my daily routine."
        "I joke about my pain so they don’t worry.",
        "I smile when I want to scream — habit now.",
        "I never say 'I’m struggling.' Just 'I’m tired.'",
        "They ask what’s wrong. I say 'Just busy.'",
        "I changed the subject — it’s safer that way.",
        "I swallow every reaction and call it maturity.",
        "No one sees the tears I blink back daily.",
        "Even when I want to open up, I hold back.",
        "Strength is my mask — and it’s glued on.",
        "They praise me for being composed. I call it hiding."        
    ],
    'label': ["Suppression"] * 29
}
df_supression=pd.DataFrame(suppression_data)
df=pd.concat([df,df_supression], ignore_index=True)

exhaustion_data = {
    'text': [
        "I'm tired in ways sleep doesn’t fix.",
        "Woke up drained and didn’t even do anything yet.",
        "I’m mentally worn out from pretending I’m okay.",
        "Every task feels like climbing a mountain.",
        "I feel slow, blurry — like I’m living underwater.",
        "I don’t even have the energy to be sad.",
        "I’m always tired, even after resting.",
        "Small talk feels like lifting weights.",
        "My body moves. My mind barely follows.",
        "I’m running on autopilot, and even that’s flickering.",
        "Every conversation feels like a chore.",
        "Even joy feels heavy right now.",
        "I’m so tired, I forgot why I started this.",
        "There’s no fight left, just surviving the day.",
        "I cancel plans because even smiling is exhausting.",
        "I’m not lazy — I’m just depleted.",
        "I sit still and feel like I’ve sprinted.",
        "Sleep is a pause, not a fix.",
        "I keep pushing, hoping I’ll recharge eventually.",
        "Exhaustion has replaced motivation."
    ],
    'label': ["Exhaustion"] * 20
}

df_exhaustion=pd.DataFrame(exhaustion_data)
df=pd.concat([df, df_exhaustion], ignore_index=True)

disappointment_data = {
    'text': [
        "They promised they'd show up — but didn’t.",
        "I hoped for more. Got silence instead.",
        "I said it mattered. They shrugged.",
        "The reply came two days later — and it was dry.",
        "My excitement met their indifference.",
        "I shared something personal. They changed the topic.",
        "I smiled, expecting kindness. Got a joke instead.",
        "I needed comfort. Got a checklist.",
        "They forgot — again.",
        "I thought this moment would feel better than it does.",
        "I believed them. I shouldn’t have.",
        "The gift felt generic. The moment felt missed.",
        "They remembered, but only after I reminded them twice.",
        "Their attention disappeared right when it mattered.",
        "I opened up — they closed off.",
        "The plan sounded perfect. The execution felt empty.",
        "Every effort I made felt unseen.",
        "I didn’t expect magic. Just honesty.",
        "I asked for support. Got advice.",
        "Their apology lacked feeling — just words, no weight."
    ],
    'label': ["Disappointment"] * 20
}

df_disappointment=pd.DataFrame(disappointment_data)
df=pd.concat([df, df_disappointment], ignore_index=True)

sadness_data = {
    'text': [
        "I feel heavy — like something inside won’t lift.",
        "I cry randomly, and I don’t know why anymore.",
        "Everything feels muted. Even joy echoes faintly.",
        "I miss people who probably forgot me.",
        "I watched them leave — quietly hurting.",
        "I wake up sad, even after good dreams.",
        "I wish I could explain what I’m feeling — but I can’t.",
        "Music helps me feel things I’ve buried.",
        "I feel alone even in loving spaces.",
        "Tears come easier than smiles now.",
        "I keep trying. But the sadness follows.",
        "I laughed yesterday — but today feels gray again.",
        "Every happy moment reminds me of something I lost.",
        "I want to be okay. I just don’t know how.",
        "Even the sun feels distant today.",
        "I wish I didn’t feel so fragile all the time.",
        "I’m not heartbroken — just quietly bruised.",
        "They say it’ll pass. But this sadness feels permanent.",
        "I’m scared I’m becoming my sadness.",
        "I smile less. Feel more. And say nothing."
        "Everything felt okay — until they said goodbye.",
        "I smiled through the ache because it was expected.",
        "She hugged me and I nearly cried — but didn’t.",
        "I watched them laugh and felt miles away.",
        "They moved on while I stayed stuck in yesterday.",
        "Every familiar place feels unfamiliar now.",
        "I wrote a message I’ll never send.",
        "My favorite song doesn’t hit the same anymore.",
        "I talked about my feelings and got silence back.",
        "I miss how it used to feel — light, free, real."     
    ],
    'label': ["Sadness"] * 29
}

df_sadness=pd.DataFrame(sadness_data)
df=pd.concat([df, df_sadness], ignore_index=True)

jealousy_data = {
    'text': [
        "They got the praise I worked so hard for — I smiled anyway.",
        "I pretend I’m happy for them, but it stings every time.",
        "I watch them thrive while I feel stuck in place.",
        "Their life looks perfect, and mine feels paused.",
        "They celebrated her loudly — no one noticed me.",
        "My success feels invisible next to theirs.",
        "He didn’t even try — and still got what I wanted.",
        "I wonder why they always shine brighter.",
        "She gets compliments for things I work twice as hard on.",
        "I cheered them on, while envying every part of it.",
        "They called her talented. I’m still waiting to be seen.",
        "I feel small when they praise everyone but me.",
        "They get invited — I get forgotten.",
        "She walked in and the room lit up. I quietly dimmed.",
        "I scroll and compare. And shrink a little each time.",
        "They don’t even notice how much I’m aching for validation.",
        "Why do their wins feel like my losses?",
        "I feel bitter, even though I don’t want to.",
        "They seem effortlessly loved. I feel replaceable.",
        "I wish I felt proud — instead, I feel behind."
    ],
    'label': ["Jealousy"] * 20
}

df_jealousy=pd.DataFrame(jealousy_data)
df=pd.concat([df, df_jealousy], ignore_index=True)

sarcasm_data = {
    'text': [
        "Oh wow, another person ghosted me — must be my lucky day.",
        "Sure, I’m fine — just crying into my cereal again.",
        "Great, another meeting where I pretend to be enthusiastic.",
        "I love feeling invisible — really adds spice to my day.",
        "Wow, that compliment felt deeply… robotic.",
        "Nothing says 'I'm loved' like a forwarded birthday text.",
        "Oh good, they finally replied — after three weeks of deep care, clearly.",
        "Fantastic — my plans fell apart again. I must be cursed with charm.",
        "Love when people ask how I am but talk over the answer.",
        "I’m SO productive today. I organized my disappointment by color.",
        "Always a joy when your pain becomes someone else's punchline.",
        "I adore the silence after asking for help — it's poetic, really.",
        "Yay, another day of pretending I slept well and care deeply.",
        "Sure, I feel supported — emotionally ghosted, but supported.",
        "How kind of them to notice I exist... once a month.",
        "Can’t wait to be comforted by inspirational quotes again.",
        "So glad everyone else is thriving. Makes being sad more efficient.",
        "Nothing like laughing at your own misery to lighten the mood.",
        "I’m basically a therapist now — for people who never ask how I’m doing.",
        "Yes, I'm absolutely glowing — it’s called sleep deprivation and emotional denial."
    ],
    'label': ["Sarcasm"] * 20
}

df_sarcasm=pd.DataFrame(sarcasm_data)
df=pd.concat([df, df_sarcasm], ignore_index=True)

hope_data = {
    'text': [
        "Even in silence, hope flickers quietly inside me.",
        "I’m still dreaming — even if the dream keeps shifting.",
        "There’s a small voice inside me saying ‘keep going.’",
        "I felt something gentle today — maybe it's hope.",
        "It’s not perfect, but it’s possible. And that’s enough for me.",
        "I saw a light today. Not bright. But mine.",
        "I don’t know what’s next, but I believe it’s worth walking toward.",
        "Something told me to try again — so I did.",
        "Hope doesn’t shout. It whispers. And I’m learning to listen.",
        "My heart feels unsure — but still open.",
        "I caught myself smiling at a ‘what if.’",
        "They said 'maybe' — and for once, it didn’t feel disappointing.",
        "I planted an idea today. Maybe something will grow.",
        "I felt peace for a moment. It was soft, but strong.",
        "There’s a future I can’t see clearly, but it calls me anyway.",
        "I’m tired, but not empty. There’s a difference.",
        "Sometimes hope looks like showing up again.",
        "I chose to believe today. It wasn’t easy. But I did.",
        "They looked at me like I mattered — maybe I do.",
        "I almost quit. But I didn’t. That counts for something."
    ],
    'label': ["Hope"] * 20
}

guilt_data = {
    'text': [
        "I hurt them with words I can’t take back.",
        "I watched their face fall — and did nothing.",
        "I lied to protect myself, but now I’m drowning in guilt.",
        "Every time they smile at me, I flinch inside.",
        "They forgave me. I can’t forgive myself.",
        "I replay that moment again and again. I should’ve done better.",
        "I caused pain, even though I didn’t mean to.",
        "They trusted me — and I let them down.",
        "I avoided them. Now I miss the chance to make things right.",
        "What I said was cruel. What I feel now is worse.",
        "They cried. I stayed silent. That silence haunts me.",
        "I disappointed someone who believed in me.",
        "I broke a promise I swore I’d keep.",
        "I made them feel small — and that wasn’t fair.",
        "Their message was kind. Mine was defensive. I regret that.",
        "I didn’t show up — and it mattered more than I realized.",
        "I keep apologizing. But the shame stays.",
        "They needed me. And I chose myself.",
        "I messed up something good — and I know it was my fault.",
        "Every ‘it’s okay’ they said felt like a dagger in disguise."
    ],
    'label': ["Guilt"] * 20
}

regret_data = {
    'text': [
        "I should’ve stayed — now I replay the goodbye.",
        "I walked away too quickly, and it changed everything.",
        "I had the chance to speak my truth — but didn’t.",
        "I let fear control the choice I’ll never undo.",
        "They reached out. I ignored it. Now it’s too late.",
        "I ruined something I didn’t know I needed.",
        "I missed the moment that could've made a difference.",
        "I chose silence when I needed to shout.",
        "I keep wondering how things would be if I had tried.",
        "I held back — and the distance became permanent.",
        "I regret making them feel unloved.",
        "I made a decision that I regret daily.",
        "If I had listened more, they might still be here.",
        "I pushed them away before they could even hurt me.",
        "I had the words ready — but swallowed them instead.",
        "We had a future, and I let it slip.",
        "I let my pride win — and lost the connection.",
        "I turned away from someone I needed.",
        "I gave up too soon — and I feel it now.",
        "My hesitation became our ending."
    ],
    'label': ["Regret"] * 20
}


betrayal_data = {
    'text': [
        "They broke the trust I built brick by brick.",
        "I gave them everything — they shared it with someone else.",
        "Their silence told me I was betrayed before the words did.",
        "They used what I shared to hurt me.",
        "I saw them comfort the person who hurt me.",
        "They promised loyalty, then vanished when I needed it most.",
        "They chose someone else without even telling me.",
        "I confided in them — they laughed behind my back.",
        "They told my secrets to the people I feared most.",
        "I was loyal. They were calculating.",
        "They protected their lie over our bond.",
        "They smiled while tearing me apart inside.",
        "They denied everything. I lost more than the truth.",
        "I was replaced without warning.",
        "They pretended to care. I pretended not to notice.",
        "Their support disappeared when others came into the room.",
        "They showed affection — then mocked me later.",
        "They covered their betrayal with affection.",
        "I realized they never planned to stay.",
        "They made me feel safe just to pull the rug away."
    ],
    'label': ["Betrayal"] * 20
}

isolation_data = {
    'text': [
        "I’m surrounded by people, and still feel completely alone.",
        "No one truly sees me, and that emptiness hurts.",
        "The room is loud. I feel invisible.",
        "I talk — and no one listens.",
        "I scroll for connection and find silence.",
        "I smiled at them. They didn’t notice.",
        "I’m there, but not really part of anything.",
        "They laughed together. I faded quietly.",
        "Everyone's close, and I feel distant.",
        "I spoke in the group chat. No one replied.",
        "I laughed to feel included, but I didn’t belong.",
        "I feel like an extra in their story.",
        "I keep showing up, but never get picked.",
        "They all have someone. I pretend I do.",
        "I exist in spaces where I’m not acknowledged.",
        "I listen to their problems, but share none of mine.",
        "I’m not part of their inside jokes.",
        "My name goes unnoticed even when I’m present.",
        "They hugged each other. I stood in the back.",
        "I keep hoping for a ‘hey, you okay?’ — it never comes."
    ],
    'label': ["Isolation"] * 20
}


numbness_data = {
    'text': [
        "I don’t feel anything anymore — just a quiet fog.",
        "I laugh when I should cry, and cry when I feel nothing.",
        "Even good news feels muted now.",
        "I move through the day, but I’m not really in it.",
        "I stare at the ceiling — waiting to feel something.",
        "Smiles don’t reach me anymore.",
        "I watch people react to life — I stay blank.",
        "Food tastes like paper. Sounds feel distant.",
        "My memories feel like someone else’s story.",
        "I fake enthusiasm. Inside, I’m static.",
        "I’m not sad. I’m just… absent.",
        "I don’t connect to anything, even the things I love.",
        "They hugged me. I felt nothing.",
        "My emotions used to shout. Now they whisper, if at all.",
        "I stopped caring about things I used to fight for.",
        "I react because I should, not because I feel.",
        "I feel paused. Not broken. Just paused.",
        "Tears won’t come. Words don’t matter.",
        "My thoughts echo in a hollow space.",
        "I want to feel something — even if it hurts."
    ],
    'label': ["Numbness"] * 20
}

sadness_vs_bittersweet = [
    ("She left without a word — now everything reminds me of her.", "Sadness"),
    ("She left with a hug and a soft smile — I cherish it, even as it hurts.", "Bittersweet"),
    
    ("I reread the old texts and cry — I miss what we had.", "Sadness"),
    ("I reread our messages with a smile — we had something beautiful once.", "Bittersweet"),
    
    ("My room feels empty without them.", "Sadness"),
    ("My room holds memories that comfort me and sting a little.", "Bittersweet")
]

guilt_vs_betrayal = [
    ("I broke their trust and I hate myself for it.", "Guilt"),
    ("They broke my trust and didn’t even flinch.", "Betrayal"),

    ("I knew I was wrong the moment I saw their reaction.", "Guilt"),
    ("They knew they were hurting me and did it anyway.", "Betrayal"),

    ("I messed up — I wish I could make it right.", "Guilt"),
    ("They messed up — and made me feel like I was the villain.", "Betrayal")
]

sarcasm_vs_jealousy = [
    ("Oh great, another group photo I’m cropped out of — classic.", "Sarcasm"),
    ("They smiled in every group photo while I stood behind the lens again.", "Jealousy"),

    ("Fantastic! They forgot me... again. I should start sending postcards from the void.", "Sarcasm"),
    ("They invited her and not me — I guess I’m just background noise.", "Jealousy"),

    ("I love being emotionally ghosted. It's my favorite hobby now.", "Sarcasm"),
    ("He comforts her the way I always wished he’d comfort me.", "Jealousy")
]

numbness_vs_suppression = [
    ("I smile, nod, go through the motions — but feel nothing underneath.", "Numbness"),
    ("I smile to keep people from asking — everything stays locked inside.", "Suppression"),

    ("Even music sounds flat lately — I’m barely present.", "Numbness"),
    ("I change the song when it hits too deep — I can’t afford to unravel.", "Suppression"),

    ("I keep showing up, empty as ever.", "Numbness"),
    ("I keep showing up, pretending nothing’s wrong.", "Suppression")
]

jealousy_vs_disappointment = [
    ("They got the credit I worked hard for — I smiled anyway, but it stung.", "Jealousy"),
    ("They forgot to mention me — again. I keep hoping they'll notice next time.", "Disappointment"),

    ("She got praised for my idea. I stayed quiet and bitter.", "Jealousy"),
    ("They said 'sorry we missed you' — I waited the whole day.", "Disappointment"),

    ("They admired her for what I whispered first. I felt erased.", "Jealousy"),
    ("They promised they'd show up. They didn’t. I still waited.", "Disappointment")
]

guilt_vs_bittersweet = [
    ("I said something cruel — they smiled anyway, and that haunts me.", "Guilt"),
    ("They hugged me before leaving — I smiled through the ache.", "Bittersweet"),

    ("I failed them when they needed me most — and I see it every time I look at them.", "Guilt"),
    ("We shared coffee and memories before she moved away. I smiled, quietly aching.", "Bittersweet"),

    ("I should’ve been there. I wasn’t. They don’t blame me — but I do.", "Guilt"),
    ("She said 'we had something special' — and I agreed, even while hurting.", "Bittersweet")
]

numbness_vs_sadness = [
    ("I’m not sad — I just don’t feel anything anymore.", "Numbness"),
    ("I cried quietly while they laughed — I missed feeling connected.", "Sadness"),

    ("I go through the motions like a ghost — untouched by joy or pain.", "Numbness"),
    ("I still wear the hoodie they left behind — and it hurts every time.", "Sadness"),

    ("People talk at me — I nod and float, disconnected.", "Numbness"),
    ("They moved on. I’m stuck in our goodbye.", "Sadness")
]

betrayal_vs_guilt = [
    ("They broke my trust and blamed me for reacting.", "Betrayal"),
    ("I broke their trust and now carry the shame silently.", "Guilt"),

    ("They smiled while leaking my secrets — I couldn’t believe it.", "Betrayal"),
    ("I gave up on them when they needed me — I still regret it deeply.", "Guilt"),

    ("They let someone else speak for me — after promising to protect my words.", "Betrayal"),
    ("They expected me to listen. I didn’t — and I know it hurt them.", "Guilt")
]

jealousy_vs_sadness = [
    ("They smiled at her like I didn’t exist. I wish it didn’t sting so much.", "Jealousy"),
    ("They don’t notice me anymore — even when I try to be seen.", "Sadness"),
    
    ("I saw her take my spot, my praise, and I clapped anyway.", "Jealousy"),
    ("They drifted away. I watched, helpless to change anything.", "Sadness"),
    
    ("He laughed with her the way I wanted him to with me.", "Jealousy"),
    ("They stopped asking how I feel — now I just carry it alone.", "Sadness")
]

bittersweet_vs_joy = [
    ("We danced like nothing had changed — but it was our last goodbye.", "Bittersweet"),
    ("We danced without worries — laughter echoed louder than music.", "Joy"),
    
    ("They hugged me tight before leaving — and I smiled through the ache.", "Bittersweet"),
    ("They hugged me out of nowhere, and I felt truly seen.", "Joy"),
    
    ("I felt loved and lonely at the same time. That moment still stays.", "Bittersweet"),
    ("I felt loved without any doubts — just warmth.", "Joy")
]

sadness_vs_guilt = [
    ("I miss them deeply — and I wish I could say everything I never did.", "Sadness"),
    ("I avoided their message when they needed me — and I regret it every day.", "Guilt"),
    
    ("Their absence weighs on me. I just wish they were still here.", "Sadness"),
    ("They asked for help. I ignored it. It wasn’t right.", "Guilt"),
    
    ("I feel the ache of everything we didn’t get to do.", "Sadness"),
    ("I made them feel small during their worst moment — and that shame stays.", "Guilt")
]

bittersweet_vs_jealousy = [
    ("We shared one last coffee, laughing quietly through the ache.", "Bittersweet"),
    ("She walked in, took the spotlight — and I smiled like it didn’t hurt.", "Jealousy"),

    ("He hugged me before leaving — soft, warm, and final.", "Bittersweet"),
    ("They praised her for what I suggested — again.", "Jealousy"),

    ("We said goodbye with a smile — joy and sorrow tangled like old friends.", "Bittersweet"),
    ("He comforted her like I was invisible.", "Jealousy")
]

joy_vs_sadness = [
    ("We danced until our cheeks hurt — nothing but pure laughter.", "Joy"),
    ("I smiled at their joy while remembering everything I lost.", "Sadness"),

    ("Today felt light — like nothing could weigh me down.", "Joy"),
    ("They laughed without me — I watched from the edge, hurting quietly.", "Sadness"),

    ("I laughed so hard I couldn’t breathe — no reason, just happiness.", "Joy"),
    ("I laughed with them, but felt hollow inside.", "Sadness")
]

jealousy_vs_joy = [
    ("They laughed with her like I was never there — and I smiled like it was fine.", "Jealousy"),
    ("They laughed with me until tears rolled down — I felt completely seen.", "Joy"),

    ("She got praised for what I quietly built — I clapped, aching inside.", "Jealousy"),
    ("They praised my work, and I felt proud — not for recognition, but for growth.", "Joy"),

    ("They posed together, radiant — I stood behind the camera pretending not to care.", "Jealousy"),
    ("We posed together, radiant — genuine happiness on both our faces.", "Joy")
]

exhaustion_vs_numbness = [
    ("I’m tired in every way — mind, heart, body. It’s like I’m carrying weight everywhere.", "Exhaustion"),
    ("I don’t feel tired — I just don’t feel anything. Even the weight seems hollow.", "Numbness"),

    ("Even sleep doesn’t help anymore — I wake up and still feel drained.", "Exhaustion"),
    ("I wake up, go through motions, and feel nothing — not even tired, just disconnected.", "Numbness"),

    ("I’m emotionally burned out — joy feels like a distant echo.", "Exhaustion"),
    ("I don’t even know what joy felt like anymore. I’m just... paused.", "Numbness")
]

disappointment_vs_guilt = [
    ("They said they’d be there — I waited, and they never came.", "Disappointment"),
    ("I told them I’d be there — and then I didn’t show up. They waited.", "Guilt"),

    ("I trusted their promise — now I wonder if it ever mattered.", "Disappointment"),
    ("I broke my promise, even though they believed in me.", "Guilt"),

    ("I tried not to expect anything — but it still stung when they forgot.", "Disappointment"),
    ("I forgot something they truly needed — and I still feel the shame.", "Guilt")
]

sarcasm_vs_sadness = [
    ("Love being the emotional trash can of the group — totally my dream job.", "Sarcasm"),
    ("I feel like everyone unloads on me, but no one asks how I’m doing.", "Sadness"),

    ("Oh great, another day of being absolutely irrelevant — my specialty.", "Sarcasm"),
    ("I feel invisible in rooms full of people — no one really hears me.", "Sadness"),

    ("I’m emotionally thriving — cried at cereal commercials and spilled my coffee in despair.", "Sarcasm"),
    ("I cried over something small — maybe I’m not okay anymore.", "Sadness")
]

isolation_vs_betrayal = [
    ("They laughed together — I was never really in the circle, just near it.", "Isolation"),
    ("They laughed about me — behind closed doors, while pretending to care in public.", "Betrayal"),

    ("I exist around them, but not with them.", "Isolation"),
    ("I shared my fears — and they made them into a joke for others.", "Betrayal"),

    ("They didn’t include me — again. I stayed quiet.", "Isolation"),
    ("They said ‘always here for you’ — then vanished when I finally opened up.", "Betrayal")
]

numbness_vs_bittersweet = [
    ("I keep showing up, answering, replying — none of it really touches me anymore.", "Numbness"),
    ("We laughed through tears during our last call — I’m grateful, and grieving.", "Bittersweet"),

    ("Even joy seems unreachable — I feel blank, not broken.", "Numbness"),
    ("She waved goodbye smiling — a soft ache I’ll always carry.", "Bittersweet"),

    ("I scroll endlessly, looking for something that moves me — still blank.", "Numbness"),
    ("We held hands like always — knowing this was goodbye.", "Bittersweet")
]

guilt_vs_disappointment = [
    ("They counted on me — and I let them down when it mattered most.", "Guilt"),
    ("I counted on them — and they didn’t show up when I needed them most.", "Disappointment"),

    ("I said I’d listen — and I didn’t. That moment keeps haunting me.", "Guilt"),
    ("They said they'd listen — and brushed me off like I was overreacting.", "Disappointment"),

    ("They forgave me for being absent — I haven’t forgiven myself yet.", "Guilt"),
    ("They forgot me in a room full of people — I pretended not to notice.", "Disappointment")
]

isolation_vs_sadness = [
    ("They all chatted around me — like I wasn’t even there.", "Isolation"),
    ("I miss how we used to talk — now every message feels hollow.", "Sadness"),

    ("I sit with them, but I’m not really with them. Just... near.", "Isolation"),
    ("I hear their laughter from the hallway — and remember when it included me.", "Sadness"),

    ("I’m the background friend — visible, not engaged.", "Isolation"),
    ("I watched them drift away — no anger, just ache.", "Sadness")
]

sadness_vs_hope = [
    ("I miss the way things used to be — and I’m not sure I’ll ever feel that joy again.", "Sadness"),
    ("I miss the way things used to be — but I believe better days will come.", "Hope"),

    ("Every corner of this house echoes with her absence.", "Sadness"),
    ("Every corner of this house reminds me I still have space to grow.", "Hope"),

    ("I feel forgotten by people who used to care.", "Sadness"),
    ("I feel distant — but I trust that connections will return, even if slowly.", "Hope")
]

joy_vs_bittersweet = [
    ("We played games all evening — nothing but laughter and pure fun.", "Joy"),
    ("We played games all evening — hiding how much we'd miss this moment.", "Bittersweet"),
    
    ("I laughed until my stomach hurt — I felt fully alive.", "Joy"),
    ("I laughed, already missing the moment before it ended.", "Bittersweet")
]

betrayal_vs_isolation = [
    ("I confided in them — and they turned my words into a public joke.", "Betrayal"),
    ("I sat in the room, unheard, invisible to everyone.", "Isolation"),

    ("They used my vulnerability against me — that was calculated.", "Betrayal"),
    ("They didn’t respond — not out of malice, just absence.", "Isolation")
]

numbness_vs_hope = [
    ("I do everything, say everything — but I feel completely blank inside.", "Numbness"),
    ("I’m not okay yet — but I know someday I will be.", "Hope"),

    ("I’m numb to both good and bad — nothing feels real.", "Numbness"),
    ("I can’t see the light now — but I believe it’s out there.", "Hope"),

    ("I wake up, move, respond — it’s like my feelings signed out weeks ago.", "Numbness"),
    ("I wake up, move, respond — and with each small step, I believe I’ll feel again.", "Hope"),

    ("Even music feels like static now — nothing gets through.", "Numbness"),
    ("Even music sounds distant — but I know the melody will mean something again someday.", "Hope")
]

determination_dataset = [
    ("I’ll keep going — even if progress feels slow, I’m moving forward.", "Determination"),
    ("They said I couldn’t — so I trained harder. Giving up isn’t an option.", "Determination"),
    ("Each failure is a lesson. I’m not quitting — I’m learning.", "Determination"),
    ("Even when I doubt myself, I refuse to let that stop me.", "Determination"),
    ("I’m exhausted, but I owe it to myself to keep trying.", "Determination"),
    ("This mountain is tough — but I wasn’t built to surrender.", "Determination"),
    ("I push through the setbacks — because my dream is worth it.", "Determination"),
    ("The path is unclear, but I know where I’m heading.", "Determination"),
    ("I show up every day, even when motivation runs dry.", "Determination"),
    ("I keep climbing because I've come too far to fall now.", "Determination"),
    ("Discomfort isn’t defeat — it’s proof I’m growing.", "Determination"),
    ("I fail. I learn. I retry. That’s how I win.", "Determination"),
    ("Even with fear in my chest, I take the next step.", "Determination"),
    ("My plans change, but my focus stays locked in.", "Determination"),
    ("The dream feels distant — but I haven’t stopped chasing.", "Determination"),
    ("I whisper ‘you’ve got this’ every time doubt creeps in.", "Determination"),
    ("I don’t wait for perfect timing. I create it.", "Determination"),
    ("This may take longer than I thought — but I’m committed.", "Determination"),
    ("My friends gave up. I didn’t. That’s my difference.", "Determination"),
    ("I could rest. Or I could finish what I started.", "Determination"),
    ("I’ll rebuild myself brick by brick if I have to.", "Determination"),
    ("Every stumble fuels my next stride.", "Determination"),
    ("Persistence is my superpower — and I won’t lose it.", "Determination"),
    ("I’m tired of doubting — so I’m choosing discipline.", "Determination"),
    ("Success isn’t luck. It’s every hard choice I didn’t run from.", "Determination")
]

determination_vs_guilt = [
    ("I made mistakes — but I’m using them as fuel to improve.", "Determination"),
    ("I made mistakes — and I keep replaying them with regret.", "Guilt"),

    ("I let someone down — now I’m driven to earn back their trust.", "Determination"),
    ("I let someone down — and I hate myself for it.", "Guilt"),

    ("I missed an opportunity — next time, I’ll be ready.", "Determination"),
    ("I missed an opportunity — and I can’t stop blaming myself.", "Guilt")
]

determination_vs_hope = [
    ("I’ll push forward whether I’m scared or unsure — quitting isn’t an option.", "Determination"),
    ("I’ll move forward slowly, trusting that things will get better.", "Hope"),

    ("I set my sights on the finish line — and I’m sprinting toward it.", "Determination"),
    ("I’m unsure of where this ends, but I believe in the journey.", "Hope"),

    ("I fight through doubt with grit — because progress needs action.", "Determination"),
    ("I wait through doubt with belief — because progress needs patience.", "Hope")
]

determination_vs_bittersweet = [
    ("I’m rebuilding everything I lost — brick by brick, with zero shortcuts.", "Determination"),
    ("I’m grateful for what I had — even if it’s gone now.", "Bittersweet"),

    ("Today stings, but I’m working harder than ever.", "Determination"),
    ("Today stings, and I smile through the ache of remembering.", "Bittersweet"),

    ("I feel the pain — and I’m using it to fuel my comeback.", "Determination"),
    ("I feel the pain — and I cherish the memories it brings.", "Bittersweet")
]

df_sadness_vs_bittersweet=pd.DataFrame(sadness_vs_bittersweet, columns=['text', 'label'])
df_guilt_vs_betrayal=pd.DataFrame(guilt_vs_betrayal, columns=['text', 'label'])
df_sarcasm_vs_jealousy=pd.DataFrame(sarcasm_vs_jealousy, columns=['text', 'label'])
df_numbness_vs_suppression=pd.DataFrame(numbness_vs_suppression, columns=['text', 'label'])
df_jealousy_vs_disappointment=pd.DataFrame(jealousy_vs_disappointment, columns=['text', 'label'])
df_guilt_vs_bittersweet=pd.DataFrame(guilt_vs_bittersweet, columns=['text', 'label'])
df_numbness_vs_sadness=pd.DataFrame(numbness_vs_sadness, columns=['text','label'])
df_betrayal_vs_guilt=pd.DataFrame(betrayal_vs_guilt, columns=['text', 'label'])
df_jealousy_vs_sadness=pd.DataFrame(jealousy_vs_sadness, columns=['text', 'label'])
df_bittersweet_vs_joy=pd.DataFrame(bittersweet_vs_joy, columns=['text', 'label'])
df_sadness_vs_guilt=pd.DataFrame(sadness_vs_guilt, columns=['text', 'label'])
df_joy_vs_sadness=pd.DataFrame(joy_vs_sadness, columns=['text', 'label'])
df_bittersweet_vs_jealousy=pd.DataFrame(bittersweet_vs_jealousy, columns=['text', 'label'])
df_jealousy_vs_joy=pd.DataFrame(jealousy_vs_joy, columns=['text', 'label'])
df_exhaustion_vs_numbness = pd.DataFrame(exhaustion_vs_numbness, columns=['text', 'label'])
df_disappointment_vs_guilt = pd.DataFrame(disappointment_vs_guilt, columns=['text', 'label'])
df_sarcasm_vs_sadness= pd.DataFrame(sarcasm_vs_sadness, columns=['text', 'label'])
df_isolaton_vs_betrayal = pd.DataFrame(isolation_vs_betrayal, columns=['text', 'label'])
df_numbness_vs_bittersweet = pd.DataFrame(numbness_vs_bittersweet, columns=['text', 'label'])
df_guilt_vs_disappointment = pd.DataFrame(guilt_vs_disappointment, columns=['text', 'label'])
df_isolation_vs_sadness = pd.DataFrame(isolation_vs_sadness, columns=['text', 'label'])
df_sadness_vs_hope = pd.DataFrame(sadness_vs_hope, columns=['text', 'label'])
df_joy_vs_bittersweet= pd.DataFrame(joy_vs_bittersweet, columns=['text', 'label'])
df_betrayal_vs_isolation= pd.DataFrame(betrayal_vs_isolation, columns=['text', 'label'])
df_numbness_vs_hope= pd.DataFrame(numbness_vs_hope, columns=['text', 'label'])
df_determination=pd.DataFrame(determination_dataset, columns=['text', 'label'])
df_determination_vs_guilt= pd.DataFrame(determination_vs_guilt, columns=['text', 'label'])
df_determination_vs_hope= pd.DataFrame(determination_vs_hope, columns=['text', 'label'])
df_determination_vs_bittersweet= pd.DataFrame(determination_vs_bittersweet, columns=['text', 'label'])
df_hope= pd.DataFrame(hope_data)
df_guilt=pd.DataFrame(guilt_data)
df_regret=pd.DataFrame(regret_data)
df_isolation=pd.DataFrame(isolation_data)
df_numbness=pd.DataFrame(numbness_data)
df_betrayal=pd.DataFrame(betrayal_data)

df=pd.concat([
    df, df_betrayal, df_hope, df_guilt, df_regret, df_isolation, df_numbness, df_sadness_vs_bittersweet,
    df_guilt_vs_betrayal,df_sarcasm_vs_jealousy, df_numbness_vs_suppression, df_jealousy_vs_disappointment,df_sadness_vs_guilt,
    df_guilt_vs_bittersweet, df_numbness_vs_sadness, df_betrayal_vs_guilt, df_jealousy_vs_sadness, df_bittersweet_vs_joy,
    df_bittersweet_vs_jealousy, df_joy_vs_sadness, df_joy_vs_sadness, df_disappointment_vs_guilt, df_exhaustion_vs_numbness,
    df_sarcasm_vs_sadness, df_isolaton_vs_betrayal, df_numbness_vs_bittersweet, df_guilt_vs_disappointment,
    df_isolation_vs_sadness, df_sadness_vs_hope, df_joy_vs_bittersweet, df_betrayal_vs_isolation, df_numbness_vs_hope,
    df_determination, df_determination_vs_guilt, df_determination_vs_hope, df_determination_vs_bittersweet
], ignore_index=True)
