# WhatsApp Group Chat Analysis with Snake AI

This guide explains how to export your WhatsApp history and use the **Snake Algorithm**—a multiclass logical classifier—to predict "who said what" using the provided Python scripts.

---

### Phase 1: Exporting Chat to your Computer

To analyze the data, you must first move the chat history from your mobile device to your computer as a `.txt` file.

#### **On iPhone (iOS)**
1. Open the WhatsApp group chat you want to analyze.
2. Tap the **Group Name** at the top to open Group Info.
3. Scroll to the bottom and tap **Export Chat**.
4. Select **Without Media** (this ensures a smaller file size and better compatibility with the scripts).
5. Choose **Save to Files** or email/AirDrop the file to your computer.
6. Extract the `.zip` file if necessary to find your `.txt` chat file.

#### **On Android**
1. Open the WhatsApp group chat.
2. Tap the **three dots (⋮)** in the top right corner.
3. Tap **More** > **Export chat**.
4. Select **Without Media**.
5. Send the file to yourself via Email, Google Drive, or USB transfer.
6. Save the file on your computer as a `.txt` file.

---

### Phase 2: System Setup

Ensure you have the following files in the same folder:
1. `groupchatfast.py`: The main execution script.
2. `algorithmeai.py`: The core logic for the Snake AI model.
3. `your_chat.txt`: Your exported WhatsApp file.

**Dependencies:**
The script requires `pandas` for data handling:
```bash
pip install pandas
```

## Phase 3: Run
```bash
python groupchatfast.py your_chat.txt
```
