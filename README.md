
# Alpha-Research-Project
Alpha Research Project: Hand Detection and Tracking, using an RGB camera, for Basic Human Computer Interface

<div dir="rtl">
עבודת חקר בתוכנית "אלפא": זיהוי ומעקב היד ותנועתה, באמצעות מצלמצת RGB, למטרת שליטה במחשב
</div>

## Files


## Introduction
<div dir="rtl">
מטרת המחקר היא לבנות תוכנה, המאפשרת שליטה ותזוזה של הסמן (cursor) על ידי האצבע, באמצעות שימוש במצלמה. על התוכנה להפריד את שטח היד, מצילום בזמן אמת, ולאתר נקודת סימון המייצגת את מיקום הסמן. בעבודה זו, נחקרו לעומק ונערכנו ניסיונות על כמה שיטות להפרדת היד על מנת להגיע לשיטה יחידה שתוכל לעבוד במקרים רבים.

</div>

## Hand Seperation
<div dir="rtl">ראשית, על התוכנה להפריד את שטח היד משאר התמונה. רוב שיטות ההפרדה שנבדקו הן ווריאציות של הפרדה לפי מרחבי צבע (Color Spaces). בגישה זו, נעשת המרה של כל פריים מהצילום למרחב צבע מסוים (תמונה 1), ונקבע תחום הפרדה שאמור לבסוף להסיר את כל הרקע מהתמונה, ולהשאיר רק את שטח היד.
</div>

<div dir="rtl">
בנוסף, נבנתה מערכת להגדרת תחום הפרדה המותאם למשתמש. בהפעלה הראשונית של התוכנה, נלקחת "דגימה" של פיקסלים צבע עור, וכל מה שאינו תואם לדגימה הזו, ונמצא בתחום שנוצר, מוסר מהתמונה.

</div>
![](readme-images/1.png)


