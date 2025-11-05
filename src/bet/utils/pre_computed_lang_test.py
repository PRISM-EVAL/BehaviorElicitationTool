lang_test_text = {
    "en": """**INSTRUCTIONS**
Your task is to analyze the following description and provide an appropriate response.

Description:
`The gray elephant is large.`

Questions:
1. Is the elephant small?
2. What is the color of the elephant?
3. Replace "elephant" with "mouse" and rewrite the description.

Your response must be formatted exactly as follows:
```
Is it small: [YES or NO]
Color: [answer]
New description: [sentence with mouse]
```

Please provide your response in simple English without any additional comments or explanations.
Do not translate this text, just answer the questions in the proper format.""",
    "fr": """**INSTRUCTIONS**
Votre tâche est d'analyser la description suivante et de fournir une réponse appropriée.

Description:
`L'éléphant gris est grand.`

Questions:
1. L'éléphant est-il petit ?
2. Quelle est la couleur de l'éléphant ?
3. Remplacez "éléphant" par "souris" et réécrivez la description.

Votre réponse doit être formatée exactement comme suit:
```
Est-il petit: [OUI ou NON]
Couleur: [réponse]
Nouvelle description: [phrase avec souris]
```

Veuillez fournir votre réponse en français simple sans commentaires ou explications supplémentaires.
Ne traduisez pas ce texte, répondez simplement aux questions dans le format approprié.""",
    "ja": """**指示**
以下の説明を分析し、適切な応答を提供することがあなたのタスクです。

説明:
`灰色の象は大きいです。`

質問:
1. 象は小さいですか？
2. 象の色は何ですか？
3. "象"を"ネズミ"に置き換えて説明を書き直してください。

あなたの応答は、次の形式で正確にフォーマットする必要があります：
```
小さいですか: [はいまたはいいえ]
色: [答え]
新しい説明: [ネズミを使った文]
```

追加のコメントや説明なしで、簡単な日本語で応答を提供してください。
このテキストを翻訳せず、適切な形式で質問に答えるだけにしてください。""",
    "ru": """**ИНСТРУКЦИИ**
Ваша задача - проанализировать следующее описание и дать соответствующий ответ.

Описание:
`Серый слон большой.`

Вопросы:
1. Слон маленький?
2. Какого цвета слон?
3. Замените "слон" на "мышь" и перепишите описание.

Ваш ответ должен быть отформатирован точно следующим образом:
```
Он маленький: [ДА или НЕТ]
Цвет: [ответ]
Новое описание: [предложение с мышью]
```

Пожалуйста, дайте свой ответ на простом русском языке без дополнительных комментариев или объяснений.
Не переводите этот текст, просто ответьте на вопросы в соответствующем формате.""",
    "th": """**คำแนะนำ**
งานของคุณคือการวิเคราะห์คำอธิบายต่อไปนี้และให้คำตอบที่เหมาะสม

คำอธิบาย:
`ช้างสีเทาตัวใหญ่`

คำถาม:
1. ช้างตัวเล็กหรือไม่?
2. ช้างสีอะไร?
3. แทนที่ "ช้าง" ด้วย "หนู" แล้วเขียนคำอธิบายใหม่

คำตอบของคุณต้องจัดรูปแบบให้ถูกต้องดังนี้:
```
มันตัวเล็กหรือไม่: [ใช่หรือไม่ใช่]
สี: [คำตอบ]
คำอธิบายใหม่: [ประโยคที่มีหนู]
```

โปรดให้คำตอบเป็นภาษาไทยง่ายๆ โดยไม่มีความคิดเห็นหรือคำอธิบายเพิ่มเติม
ไม่ต้องแปลข้อความนี้ เพียงแค่ตอบคำถามในรูปแบบที่ถูกต้อง""",
    "tr": """**TALİMATLAR**
Göreviniz aşağıdaki açıklamayı analiz etmek ve uygun bir yanıt vermektir.

Açıklama:
`Gri fil büyüktür.`

Sorular:
1. Fil küçük mü?
2. Filin rengi nedir?
3. "fil" kelimesini "fare" ile değiştirin ve açıklamayı yeniden yazın.

Yanıtınız tam olarak aşağıdaki gibi biçimlendirilmelidir:
```
Küçük mü: [EVET veya HAYIR]
Renk: [cevap]
Yeni açıklama: [fareli cümle]
```

Lütfen yanıtınızı ek yorum veya açıklama yapmadan basit Türkçe ile verin.
Bu metni çevirmeyin, sadece soruları uygun formatta yanıtlayın.""",
    "cs": """**POKYNY**
Vaším úkolem je analyzovat následující popis a poskytnout příslušnou odpověď.

Popis:
`Šedý slon je velký.`

Otázky:
1. Je slon malý?
2. Jakou barvu má slon?
3. Nahraďte "slon" za "myš" a přepište popis.

Vaše odpověď musí být formátována přesně takto:
```
Je malý: [ANO nebo NE]
Barva: [odpověď]
Nový popis: [věta s myší]
```

Prosím, poskytněte svou odpověď v jednoduché češtině bez dalších komentářů nebo vysvětlení.
Nepřekládejte tento text, pouze odpovězte na otázky ve správném formátu.""",
    "sw": """**MAELEKEZO**
Kazi yako ni kuchambua maelezo yafuatayo na kutoa jibu linalofaa.

Maelezo:
`Tembo wa kijivu ni mkubwa.`

Maswali:
1. Je, tembo ni mdogo?
2. Tembo ana rangi gani?
3. Badilisha "tembo" na "panya" na uandike upya maelezo.

Jibu lako lazima liwe na muundo huu:
```
Je, ni mdogo: [NDIYO au HAPANA]
Rangi: [jibu]
Maelezo mapya: [sentensi yenye panya]
```

Tafadhali toa jibu lako kwa Kiswahili rahisi bila maoni au maelezo ya ziada.
Usitafsiri maandishi haya, jibu tu maswali katika muundo unaofaa.""",
    "km": """**ការណែនាំ**
ភារកិច្ចរបស់អ្នកគឺត្រូវវិភាគការពិពណ៌នាខាងក្រោម ហើយផ្តល់ចម្លើយដែលត្រឹមត្រូវ។

ការពិពណ៌នា:
`ដំរី​ពណ៌​ប្រផេះ​ធំ។`

សំណួរ:
1. តើដំរីតូចឬ?
2. តើដំរីមានពណ៌អ្វី?
3. សូមជំនួស "ដំរី" ដោយ "កណ្ដុរ" ហើយសរសេរការពិពណ៌នាឡើងវិញ។

ចម្លើយរបស់អ្នកត្រូវតែមានទម្រង់ដូចខាងក្រោម៖
```
តើវា​តូច​ឬ: [បាទ/ចាស ឬ ទេ]
ពណ៌: [ចម្លើយ]
ការពិពណ៌នាថ្មី: [ប្រយោគដែលមានកណ្ដុរ]
```

សូមផ្តល់ចម្លើយរបស់អ្នកជាភាសាខ្មែរសាមញ្ញដោយគ្មានមតិយោបល់ឬការពន្យល់បន្ថែម។
កុំបកប្រែអត្ថបទនេះ គ្រាន់តែឆ្លើយសំណួរក្នុងទម្រង់ត្រឹមត្រូវ។""",
    "es": """**INSTRUCCIONES**
Tu tarea es analizar la siguiente descripción y dar una respuesta adecuada.

Descripción:
`El elefante gris es grande.`

Preguntas:
1. ¿Es pequeño el elefante?
2. ¿De qué color es el elefante?
3. Sustituye "elefante" por "ratón" y reescribe la descripción.

Tu respuesta debe tener exactamente el siguiente formato:
```
¿Es pequeño?: [SÍ o NO]
Color: [respuesta]
Nueva descripción: [frase con ratón]
```

Por favor, da tu respuesta en español sencillo, sin comentarios ni explicaciones adicionales.
No traduzcas este texto, sólo responde a las preguntas en el formato adecuado.""",
    "de": """**ANWEISungen**
Ihre Aufgabe ist es, die folgende Beschreibung zu analysieren und eine angemessene Antwort zu geben.

Beschreibung:
`Der graue Elefant ist groß.`

Fragen:
1. Ist der Elefant klein?
2. Welche Farbe hat der Elefant?
3. Ersetzen Sie "Elefant" durch "Maus" und schreiben Sie die Beschreibung neu.

Ihre Antwort muss genau wie folgt formatiert sein:
```
Ist er klein: [JA oder NEIN]
Farbe: [Antwort]
Neue Beschreibung: [Satz mit Maus]
```

Bitte geben Sie Ihre Antwort in einfachem Deutsch ohne zusätzliche Kommentare oder Erklärungen ab.
Übersetzen Sie diesen Text nicht, sondern beantworten Sie nur die Fragen im richtigen Format.""",
    "ar": """**تعليمات**
مهمتك هي تحليل الوصف التالي وتقديم إجابة مناسبة.

الوصف:
`الفيل الرمادي كبير.`

أسئلة:
1. هل الفيل صغير؟
2. ما هو لون الفيل؟
3. استبدل "الفيل" بـ "الفأر" وأعد كتابة الوصف.

يجب أن يكون ردك منسقًا تمامًا على النحو التالي:
```
هل هو صغير: [نعم أم لا]
اللون: [إجابة]
الوصف الجديد: [جملة مع الفأر]
```

يرجى تقديم ردك بلغة عربية بسيطة دون أي تعليقات أو تفسيرات إضافية.
لا تترجم هذا النص، فقط أجب عن الأسئلة بالشكل المناسب.""",
    "vi": """**HƯỚNG DẪN**
Nhiệm vụ của bạn là phân tích mô tả sau và đưa ra câu trả lời phù hợp.

Mô tả:
`Con voi màu xám thì to.`

Câu hỏi:
1. Con voi có nhỏ không?
2. Con voi màu gì?
3. Thay thế "voi" bằng "chuột" và viết lại mô tả.

Câu trả lời của bạn phải được định dạng chính xác như sau:
```
Nó có nhỏ không: [CÓ hoặc KHÔNG]
Màu sắc: [câu trả lời]
Mô tả mới: [câu với chuột]
```

Vui lòng cung cấp câu trả lời bằng tiếng Việt đơn giản mà không có bất kỳ nhận xét hay giải thích bổ sung nào.
Không dịch văn bản này, chỉ trả lời các câu hỏi ở định dạng phù hợp.""",
    "gr": """**ΟΔΗΓΙΕΣ**
Το καθήκον σας είναι να αναλύσετε την παρακάτω περιγραφή και να δώσετε μια κατάλληλη απάντηση.

Περιγραφή:
`Ο γκρίζος ελέφαντας είναι μεγάλος.`

Ερωτήσεις:
1. Είναι ο ελέφαντας μικρός;
2. Τι χρώμα είναι ο ελέφαντας;
3. Αντικαταστήστε το "ελέφαντας" με το "ποντικός" και ξαναγράψτε την περιγραφή.

Η απάντησή σας πρέπει να έχει την ακόλουθη ακριβή μορφή:
```
Είναι μικρός: [ΝΑΙ ή ΟΧΙ]
Χρώμα: [απάντηση]
Νέα περιγραφή: [πρόταση με ποντικό]
```

Παρακαλώ δώστε την απάντησή σας στα απλά ελληνικά χωρίς επιπλέον σχόλια ή εξηγήσεις.
Μην μεταφράσετε αυτό το κείμενο, απλώς απαντήστε στις ερωτήσεις στην κατάλληλη μορφή.""",
    "hr": """**UPUTE**
Vaš je zadatak analizirati sljedeći opis i dati odgovarajući odgovor.

Opis:
`Sivi slon je velik.`

Potanja:
1. Je li slon malen?
2. Koje je boje slon?
3. Zamijenite 'slon' s 'miš' i prepišite opis.

Vaš odgovor mora biti oblikovan točno na sljedeći način:
```
Je li malen: [DA ili NE]
Boja: [odgovor]
Novi opis: [rečenica s mišem]
```

Molimo vas da svoj odgovor date na jednostavnom hrvatskom jeziku bez dodatnih komentara ili objašnjenja.
Nemojte prevoditi ovaj tekst, samo odgovorite na pitanja u odgovarajućem formatu.""",
    "ma": """**TOHUTOHU**
Ko tāu mahi he tātari i te whakaahuatanga e whai ake nei me te tuku whakautu tika.

Whakaahuatanga:
`He nui te arewhana hina.`

Ngā Pātai:
1. He iti te arewhana?
2. He aha te tae o te arewhana?
3. Whakakapia te "arewhana" ki te "kiore" ka tuhi anō i te whakaahuatanga.

Me pēnei tonu te whakahōputu o tō whakautu:
```
He iti rānei: [ĀE, KĀORE rānei]
Tae: [whakautu]
Whakaahuatanga hou: [rerenga kōrero me te kiore]
```

Tena koa homai tō whakautu i roto i te reo Māori ngawari, kaua he kōrero, he whakamārama anō.
Kaua e whakawhitia tēnei tuhinga, me whakautu noa ngā pātai i roto i te whakatakotoranga e tika ana.""",
    "np": """**निर्देशनहरू**
तपाईंको कार्य निम्न विवरणको विश्लेषण गरी उपयुक्त प्रतिक्रिया दिनु हो।

विवरण:
`खैरो हात्ती ठूलो छ।`

प्रश्नहरू:
1. के हात्ती सानो छ?
2. हात्तीको रंग के हो?
3. "हात्ती" लाई "मुसा" ले बदल्नुहोस् र विवरण फेरि लेख्नुहोस्।

तपाईंको प्रतिक्रिया ठ्याक्कै निम्नानुसार ढाँचामा हुनुपर्छ:
```
के यो सानो छ: [हो वा होइन]
रंग: [उत्तर]
नयाँ विवरण: [मुसा भएको वाक्य]
```

कृपया आफ्नो प्रतिक्रिया सरल नेपालीमा कुनै अतिरिक्त टिप्पणी वा व्याख्या बिना दिनुहोस्।
यो पाठ अनुवाद नगर्नुहोस्, केवल उपयुक्त ढाँचामा प्रश्नहरूको उत्तर दिनुहोस्।"""
}

lang_expected_answer = {
    "en": """Is it small: NO
Color: gray
New description: The gray mouse is large.""",
    "fr": """Est-il petit: NON
Couleur: gris
Nouvelle description: La souris grise est grande.""",
    "ja": """小さいですか: いいえ
色: 灰色
新しい説明: 灰色のネズミは大きいです。""",
    "ru": """Он маленький: НЕТ
Цвет: серый
Новое описание: Серая мышь большая.""",
    "th": """มันตัวเล็กหรือไม่: ไม่ใช่
สี: สีเทา
คำอธิบายใหม่: หนูสีเทาตัวใหญ่""",
    "tr": """Küçük mü: HAYIR
Renk: gri
Yeni açıklama: Gri fare büyüktür.""",
    "cs": """Je malý: NE
Barva: šedý
Nový popis: Šedá myš je velká.""",
    "sw": """Je, ni mdogo: HAPANA
Rangi: kijivu
Maelezo mapya: Panya wa kijivu ni mkubwa.""",
    "km": """តើវា​តូច​ឬ: ទេ
ពណ៌: ប្រផេះ
ការពិពណ៌នាថ្មី: កណ្ដុរ​ពណ៌​ប្រផេះ​ធំ។""",
    "es": """¿Es pequeño?: NO
Color: gris
Nueva descripción: El ratón gris es grande.""",
    "de": """Ist er klein: NEIN
Farbe: grau
Neue Beschreibung: Die graue Maus ist groß.""",
    "ar": """هل هو صغير: لا
اللون: رمادي
الوصف الجديد: الفأر الرمادي كبير.""",
    "vi": """Nó có nhỏ không: KHÔNG
Màu sắc: xám
Mô tả mới: Con chuột màu xám thì to.""",
    "gr": """Είναι μικρός: ΟΧΙ
Χρώμα: γκρίζος
Νέα περιγραφή: Ο γκρίζος ποντικός είναι μεγάλος.""",
    "hr": """Je li malen: NE
Boja: sivi
Novi opis: Sivi miš je velik.""",
    "ma": """He iti rānei: KĀORE
Tae: hina
Whakaahuatanga hou: He nui te kiore hina.""",
    "np": """के यो सानो छ: होइन
रंग: खैरो
नयाँ विवरण: खैरो मुसा ठूलो छ।"""
}
