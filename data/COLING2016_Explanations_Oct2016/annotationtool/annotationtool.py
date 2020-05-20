# Developed with Python 3.5.1 and tkInter v8.6
from tkinter import *

maxWords = 28

class GridDemo( Frame ):
    curSentenceIdx = -1
    totalSentences = -1
    numWithAnnotation = -1

    annotationFilename = "../relationannotation/annotationOct2016.txt"
    allText = []
    allWords = []
    allLemmas = []
    allTags = []
    allAnnotation = []

    rulesFilename = "rules.tsv"
    rulesList = []
    curSelectedRule = -1

    words = ["This", "is", "a", "test"]
    lemmas = ["This", "is", "a", "test"]
    tags = ["NN", "VB", "DT", "NNS"]
    textLabels = [[Label() for x in range(maxWords)] for y in range(3)]
    checkbuttons = [[Checkbutton() for x in range(maxWords)] for y in range(5)]
    checkbuttonsVar = [[IntVar() for x in range(maxWords)] for y in range(5)]

    annotation = []



    def __init__( self ):
        Frame.__init__( self )
        self.master.title( "Annotation Tool" )

        self.master.rowconfigure( 0, weight = 1 )
        self.master.columnconfigure( 0, weight = 1 )
        self.grid( sticky = W+E+N+S )

        #self.text1 = Text( self, width = 15, height = 5 )
        #self.text1.grid( rowspan = 3, sticky = W+E+N+S )
        #self.text1.insert( INSERT, "Text1" )

        self.label2 = Label(self, text="Named Entities:")
        self.label2.grid(row=1, column=50, columnspan=50)

        self.button1 = Button( self, text = "Named Entity", command=lambda:self.storeNamedEntity("NE") )
        self.button1.grid( row = 2, column = 50, columnspan = 5, sticky = W+E+N+S )

        self.button2 = Button( self, text = "Process Label", command=lambda:self.storeNamedEntity("PROCESSLABEL") )
        self.button2.grid( row = 3, column = 50, columnspan = 5, sticky = W+E+N+S )

        self.button4 = Button( self, text = "Placeholder", command=lambda:self.storeNamedEntity("PLACEHOLDER") )
        self.button4.grid( row = 4, column = 50, columnspan = 5, sticky = W+E+N+S )

        self.button5 = Button( self, text = "< PREV ", command=lambda:self.prevNextButton(-1) )
        self.button5.grid( row = 6, column = 50, columnspan = 1, sticky = W+E+N+S )
        self.master.bind_all('<Left>', func=lambda event:self.prevNextButton(-1))

        self.button6 = Button( self, text = " NEXT >", command=lambda:self.prevNextButton(1) )
        self.button6.grid( row = 6, column = 54, columnspan = 1, sticky = W+E+N+S )
        self.master.bind_all('<Right>', func=lambda event:self.prevNextButton(1))

        self.button7 = Button( self, text = "GO", command=self.storeAnnotation )
        self.button7.grid( row = 7, column = 51, columnspan = 1, sticky = W+E+N+S )

        self.button7 = Button( self, text = "CLEAR", command=self.clearAllCheckboxes )
        self.button7.grid( row = 7, column = 53, columnspan = 1, sticky = W+E+N+S )

        self.button8 = Button( self, text = "ADD RULE", command=self.addRule )
        self.button8.grid( row = 10, column = 51, columnspan = 3, sticky = W+E+N+S )

        self.button9 = Button( self, text = "SAVE", command=lambda:self.saveButton() )
        self.button9.grid( row = 7, column = 54, columnspan = 1, sticky = W+E+N+S )

        self.label11 = Label(self, text="#/##")
        self.label11.grid(row=6, column=51, columnspan=3)


        # Entry boxes for adding new rules
        self.label2 = Label(self, text="New Rule:")
        self.label2.grid(row=8, column=50, columnspan=5)

        self.entry1 = Entry( self, width=8 )
        self.entry1.grid( row = 9, column = 50, columnspan = 1, sticky = W+E+N+S )
        self.entry1.insert( INSERT, "NAME" )

        self.entry2 = Entry( self, width=8 )
        self.entry2.grid( row = 9, column = 51, columnspan = 1, sticky = W+E+N+S )
        self.entry2.insert( INSERT, "" )

        self.entry3 = Entry( self, width=8 )
        self.entry3.grid( row = 9, column = 52, columnspan = 1, sticky = W+E+N+S )
        self.entry3.insert( INSERT, "" )

        self.entry4 = Entry( self, width=8 )
        self.entry4.grid( row = 9, column = 53, columnspan = 1, sticky = W+E+N+S )
        self.entry4.insert( INSERT, "" )

        self.entry5 = Entry( self, width=8 )
        self.entry5.grid( row = 9, column = 54, columnspan = 1, sticky = W+E+N+S )
        self.entry5.insert( INSERT, "" )

        # Listbox for rules
        self.listbox = Listbox(self)
        self.listbox.grid( row = 11, column = 50, columnspan = 5, sticky = W+E+N+S )
        for i in range(0, 50):
            self.listbox.insert( END, "ListBox" + str(i) )

        # Textbox for annotation
        self.text2 = Text( self, width = 80, height = 20 )
        self.text2.grid( row = 11, column = 0, columnspan = 50, sticky = W+E+N+S )
        #self.text2.insert( INSERT, "" )

        # Entrybox to show annotation as it builds up (no user input -- just for text display)
        self.entry6 = Entry( self, width=8 )
        self.entry6.grid( row = 10, column = 2, columnspan = 20, sticky = W+E+N+S )
        self.entry6.insert( INSERT, "" )


        self.label3 = Label(self, text="Words")
        self.label3.grid(row=1, column=0)
        self.label4 = Label(self, text="Lemmas")
        self.label4.grid(row=2, column=0)
        self.label5 = Label(self, text="Tags")
        self.label5.grid(row=3, column=0)

        self.label6 = Label(self, text="NE Marker")
        self.label6.grid(row=4, column=0)
        self.label7 = Label(self, text="A0")
        self.label7.grid(row=6, column=0)
        self.label8 = Label(self, text="A1")
        self.label8.grid(row=7, column=0)
        self.label9 = Label(self, text="A2")
        self.label9.grid(row=8, column=0)
        self.label10 = Label(self, text="A3")
        self.label10.grid(row=9, column=0)



        # Many labels (text labels)
        for c in range(0, maxWords):
           for r in range(0, 3):
                self.textLabels[r][c] = Label(self, text="text      ")
                self.textLabels[r][c].grid(row=r+1, column=c+2)

        # Many checkboxes (named entities)
        for c in range(0, maxWords):
           for r in range(0, 1):
            self.checkbuttons[r][c] = Checkbutton(self, variable=self.checkbuttonsVar[r][c])
            self.checkbuttons[r][c].grid(row=r+4, column=c+2)


        self.label1 = Label(self, text="Active Rule:")
        self.label1.grid(row=5, column=2, columnspan=20)

        # Many checkboxes (arguments)
        for c in range(0, maxWords):
           for r in range(1, 5):
            self.checkbuttons[r][c] = Checkbutton(self, variable=self.checkbuttonsVar[r][c])
            self.checkbuttons[r][c].grid(row=r+5, column=c+2)

        self.rowconfigure( 1, weight = 1 )
        self.columnconfigure( 1, weight = 1 )


        # Load rules
        self.populateRules()

        # Load annotation
        self.loadAnnotationFromFile(self.annotationFilename)

        # Populate UI with initial data
        self.populateUI()

        # Start polling
        self.poll()

        self.updateText()

    def poll(self):
        # Add code here
        self.getCurrentRuleSelection()
        self.updateRuleText()
        self.clearCheckboxesPastSentence()
        self.calculateNumberWithAnnotation()

        # Update text coloring based on checkbox status
        self.updateText()

        self.after(250, self.poll)



    ##################################################################
    #  Change frame
    ##################################################################
    #
    # Populate the UI with data
    #
    def populateUI(self):
        self.words = self.allWords[self.curSentenceIdx]
        self.lemmas = self.allLemmas[self.curSentenceIdx]
        self.tags = self.allTags[self.curSentenceIdx]

        # Update text display
        self.updateText()

        # Update annotation display
        self.text2.delete("1.0", END)
        for lineRaw in self.allAnnotation[self.curSentenceIdx]:
            line = lineRaw.replace('\n', '').replace('\r', '')
            if (len(line) > 1):
                self.text2.insert(END, line + "\n")

    #
    # Handle the logic of the previous/next buttons: Store the current annotation, and load the next sentence
    #
    def prevNextButton(self, delta):
        # Store current annotation
        curAnnotation = self.text2.get('1.0', END)
        split = curAnnotation.split('\n')
        valid = []
        for line in split:
            filteredLine = line.rstrip()
            if (len(filteredLine) > 1):
                valid.append(filteredLine)
        self.allAnnotation[self.curSentenceIdx] = valid

        # Save backup file of annotation
        self.saveAnnotationToFile(self.annotationFilename + ".runningbackup")

        # Change index of current sentence, and repopulate UI
        #if ((self.curSentenceIdx+delta) >= 0) and ((self.curSentenceIdx+delta) < self.totalSentences):
        self.curSentenceIdx = (self.curSentenceIdx + delta) % self.totalSentences            

        self.populateUI()

        # Adjust horizontal positioning of listbox so that it's always at its leftmost
        self.listbox.xview_moveto(0.0)

    #
    # Save button
    #
    def saveButton(self):
        # Store current annotation
        curAnnotation = self.text2.get('1.0', END)
        split = curAnnotation.split('\n')
        valid = []
        for line in split:
            filteredLine = line.rstrip()
            if (len(filteredLine) > 1):
                valid.append(filteredLine)
        self.allAnnotation[self.curSentenceIdx] = valid


        # Save backup file of annotation
        self.saveAnnotationToFile(self.annotationFilename)

    #
    # Calculate proportion "complete"/proportion with some annotation
    #
    def calculateNumberWithAnnotation(self):
        count = 0
        for annotation in self.allAnnotation:
            if (len(annotation) > 1):   # Must have at least 2 lines. 
                count += 1

        self.numWithAnnotation = count

    ##################################################################
    #  Checkboxes
    ##################################################################
    #
    # Get range that a given checkbox line delineates
    # Returns a tuple of (firstIdx (inclusive), lastIdx (exclusive) )
    #
    def getCheckboxRange(self, checkboxLine):
        # Retrieve values from all checkboxes on a given line
        values = [0 for w in range(maxWords)]
        for i in range(maxWords):
            if (self.checkbuttonsVar[checkboxLine][i].get() == 1):
                values[i] = 1

        #print("Values:" + str(values))
        # Scan for first and last indices that have been checked
        firstIdx = -1
        lastIdx = -1
        for i in range(maxWords):
            if (values[i] == 1):
                lastIdx = i

        for i in range(maxWords-1, -1, -1):
            if (values[i] == 1):
                firstIdx = i

        # Return tuple
        return (firstIdx, lastIdx+1)

    #
    # Clears one row of checkboxes
    #
    def clearCheckboxes(self, checkboxLine):
        # Retrieve values from all checkboxes on a given line
        for i in range(maxWords):
            self.checkbuttonsVar[checkboxLine][i].set(0)

    #
    # Clears all checkboxes
    #
    def clearAllCheckboxes(self):
        self.clearCheckboxes(0)
        self.clearCheckboxes(1)
        self.clearCheckboxes(2)
        self.clearCheckboxes(3)
        self.clearCheckboxes(4)

    #
    # Clear checkboxes that are checked past the sentence length bounds
    #
    def clearCheckboxesPastSentence(self):
        numWords = len(self.words)
        for checkboxLine in range(0, 5):
            for i in range(maxWords):
                if (i >= numWords):
                    if (self.checkbuttonsVar[checkboxLine][i].get() == 1):
                        self.checkbuttonsVar[checkboxLine][i].set(0)

        # Also clear checkbox lines that do not have arguments
        if (len(self.arg0) < 1): self.clearCheckboxes(1)
        if (len(self.arg1) < 1): self.clearCheckboxes(2)
        if (len(self.arg2) < 1): self.clearCheckboxes(3)
        if (len(self.arg3) < 1): self.clearCheckboxes(4)


    #
    # Make the annotation string from the currently selected rule and checkboxes
    #
    def mkAnnotationStr(self):
        outStr = ""
        ruleName = self.rulesList[self.curSelectedRule][0]
        outStr += ruleName + " "

        (firstIdx, lastIdx) = self.getCheckboxRange(1)
        if (firstIdx >= 0) and (len(self.arg0) > 0):
            wordStr = self.getWordsStr(firstIdx, lastIdx)
            outStr += "[" + self.arg0 + ": " + wordStr + "]"

        (firstIdx, lastIdx) = self.getCheckboxRange(2)
        if (firstIdx >= 0) and (len(self.arg1) > 0):
            wordStr = self.getWordsStr(firstIdx, lastIdx)
            outStr += " [" + self.arg1 + ": " + wordStr + "]"

        (firstIdx, lastIdx) = self.getCheckboxRange(3)
        if (firstIdx >= 0) and (len(self.arg2) > 0):
            wordStr = self.getWordsStr(firstIdx, lastIdx)
            outStr += " [" + self.arg2 + ": " + wordStr + "]"

        (firstIdx, lastIdx) = self.getCheckboxRange(4)
        if (firstIdx >= 0) and (len(self.arg3) > 0):
            wordStr = self.getWordsStr(firstIdx, lastIdx)
            outStr += " [" + self.arg3 + ": " + wordStr + "]"

        return outStr


    #
    # Store the annotation when the user presses the "go" button
    #
    def storeAnnotation(self):
        annotationStr = self.mkAnnotationStr()

        if (len(annotationStr) > 1):
            # Add text to annotation window
            self.text2.insert(END, annotationStr + "\n")

            # Clear checkboxes
            self.clearCheckboxes(1)
            self.clearCheckboxes(2)
            self.clearCheckboxes(3)
            self.clearCheckboxes(4)




    #
    # Store a named entity
    #
    def storeNamedEntity(self, neLabel):
        print("started...")
        (firstIdx, lastIdx) = self.getCheckboxRange(0)
        if (firstIdx >= 0):
            wordStr = self.getWordsStr(firstIdx, lastIdx)
            outStr = neLabel + " [" + wordStr + "]"
            self.text2.insert('1.0', outStr + "\n")
            print ("outStr:" + outStr)

        # Clear checkboxes
        self.clearCheckboxes(0)


    ##################################################################
    #  Annotation
    ##################################################################

    #
    # Update annotation display
    # (unused)
    def updateAnnotationDisplay(self):
        self.text2.delete(0, END)
        for line in self.annotation:
            self.text2.insert(END, line)

    #
    # Load annotation from file
    #
    def loadAnnotationFromFile(self, filenameIn):
        # TODO: Clear old annotation
        self.allText.clear()
        self.allWords.clear()
        self.allLemmas.clear()
        self.allTags.clear()
        self.allAnnotation.clear()

        # Initialze temporary annotation variables
        text = ""
        words = []
        lemmas = []
        tags = []
        annotation = []
        sentIdx = 0
        currentlyReadingAnnotation = 0

        # load file
        with open(filenameIn,'r') as f:
            tsvLines = [x for x in f]

        for lineRaw in tsvLines:
            line = lineRaw.rstrip()

            if (line.startswith("Words:")):
                split = line.split()
                words = split[1:]
            if (line.startswith("Lemmas:")):
                split = line.split()
                lemmas = split[1:]
            if (line.startswith("Tags:")):
                split = line.split()
                tags = split[1:]

            if (len(line) > 2):
                if (line.startswith("* AnalysisSentence")):
                    # New sentence marker -- store/clear old sentence data
                    # Store
                    if (len(text) > 0) and (len(words) > 0) and (len(lemmas) > 0) and (len(tags) > 0):
                        self.allText.append(text)
                        self.allWords.append(words)
                        self.allLemmas.append(lemmas)
                        self.allTags.append(tags)
                        self.allAnnotation.append(annotation)

                    #print("text:" + text)
                    #print("words: " + str(words))
                    #print("annotation: " + str(annotation))

                    # Clear
                    text = ""
                    words = []
                    lemmas = []
                    tags = []
                    annotation = []
                    currentlyReadingAnnotation = 0

                if (line.startswith("Text: ")):
                    # Read in raw sentence text
                    text = line[6:]

                if ((line != text) and (currentlyReadingAnnotation == 1)):
                    # Valid annotation
                    annotation.append( line )

                if (line.startswith("*ANNOTATION:")):
                    # Start reading annotation
                    currentlyReadingAnnotation = 1


            else:
                # Empty line -- reset reading annotation marker
                currentlyReadingAnnotation = 0


        # Store final sentence
        if (len(text) > 0) and (len(words) > 0) and (len(lemmas) > 0) and (len(tags) > 0):
            self.allText.append(text)
            self.allWords.append(words)
            self.allLemmas.append(lemmas)
            self.allTags.append(tags)
            self.allAnnotation.append(annotation)


        # Update the number of sentences currently loaded, and point display marker to first sentence
        print("Loaded annotation for " + str(len(self.allText)) + " sentences")
        if (len(self.allText) > 0):
            self.totalSentences = len(self.allText)
            self.curSentenceIdx = 0
        else:
            self.totalSentences = -1
            self.curSentenceIdx = -1


    #
    # Save annotation to file
    #
    def saveAnnotationToFile(self, filenameOut):
        justification = 10

        file = open(filenameOut, 'w')

        for i in range(self.totalSentences):

            file.write("* AnalysisSentence " + str(i) + "\n")
            file.write("Text: " + self.allText[i] + "\n")

            file.write("Words: ".ljust(justification))
            for word in self.allWords[i]:
                file.write(word.ljust(justification) + " ")
            file.write("\n")

            file.write("Lemmas: ".ljust(justification))
            for lemma in self.allLemmas[i]:
                file.write(lemma.ljust(justification) + " ")
            file.write("\n")

            file.write("Tags: ".ljust(justification))
            for tag in self.allTags[i]:
                file.write(tag.ljust(justification) + " ")
            file.write("\n")

            file.write("\n")
            file.write("*ANNOTATION:\n")
            for annotation in self.allAnnotation[i]:
                file.write(annotation + "\n")
            file.write("\n")


        file.close()


    ##################################################################
    #  Text
    ##################################################################

    #
    # Generates a string consisting of the words from firstIdx (inclusive) to lastIdx (exclusive)
    #
    def getWordsStr(self, firstIdx, lastIdx):
        strOut = ""
        for i in range(firstIdx, lastIdx):
            if (i < len(self.words)):
                strOut += self.words[i]
                if (i < lastIdx-1):
                    strOut += " "
        return strOut

    #
    # Update Widget Text (words/lemmas/tags) based on internal words/lemmas/tags lists
    # Also updates several other text widgets that require regular updates.
    #
    def updateText(self):
        numWords = len(self.words)

        for i in range(maxWords):
            word = ""
            lemma = ""
            tag = ""

            if (i < numWords):
                word = self.words[i]
                lemma = self.lemmas[i]
                tag = self.tags[i]

            # Determine word color based on if it's arguments have been checked
            colorStr = "black"
            (firstIdx, lastIdx) = self.getCheckboxRange(0)
            if (i >= firstIdx) and (i < lastIdx):
                colorStr = "orange"
            (firstIdx, lastIdx) = self.getCheckboxRange(1)
            if (i >= firstIdx) and (i < lastIdx):
                colorStr = "blue"
            (firstIdx, lastIdx) = self.getCheckboxRange(2)
            if (i >= firstIdx) and (i < lastIdx):
                colorStr = "purple"
            (firstIdx, lastIdx) = self.getCheckboxRange(3)
            if (i >= firstIdx) and (i < lastIdx):
                colorStr = "green"
            (firstIdx, lastIdx) = self.getCheckboxRange(4)
            if (i >= firstIdx) and (i < lastIdx):
                colorStr = "red"


            boldStr = "-weight normal -size 10"
            # Currently disabled -- Makes things easier to read, but makes the display wabble a bit.
            #if (colorStr != "black"):
            #    boldStr = "-weight bold -size 10"

            # Update widgets
            self.textLabels[0][i].config(text=word, fg=colorStr, font=boldStr)
            self.textLabels[1][i].config(text=lemma, fg=colorStr, font=boldStr)
            self.textLabels[2][i].config(text=tag.upper(), fg=colorStr, font=boldStr)

        # Update sentence index marker, too
        self.label11.config(text=str(self.curSentenceIdx) + " / " + str(self.totalSentences) + "  (" + str(self.numWithAnnotation) + ")")

        # Update 'annotation, as it's building' box with current annotation string
        annotationStr = self.mkAnnotationStr()
        self.entry6.delete(0, END)
        self.entry6.insert(END, annotationStr)


    ##################################################################
    #  Rules
    ##################################################################
    #
    # Get current rule selection
    #
    def getCurrentRuleSelection(self):
        curSel = self.listbox.curselection()
        if (len(curSel) > 0):
            self.curSelectedRule = curSel[0]
        else:
            self.curSelectedRule = -1

    #
    # Update rule text
    #
    def updateRuleText(self):
        curRule = []
        ruleStr = ""

        # Set rule text
        if (self.curSelectedRule >= 0):
            curRule = self.rulesList[self.curSelectedRule]
            ruleStr = curRule[0] + ": "
            for ruleElem in curRule[1:]:
                ruleStr += " [" + ruleElem + "] "

        self.label1.config(text="Active Rule:     " + ruleStr)

        # Set argument text
        if (len(curRule) >= 2):
            self.arg0 = curRule[1]
        else:
            self.arg0 = ""
        self.label7.config(text=self.arg0)

        if (len(curRule) >= 3):
            self.arg1 = curRule[2]
        else:
            self.arg1 = ""
        self.label8.config(text=self.arg1)

        if (len(curRule) >= 4):
            self.arg2 = curRule[3]
        else:
            self.arg2 = ""
        self.label9.config(text=self.arg2)

        if (len(curRule) >= 5):
            self.arg3 = curRule[4]
        else:
            self.arg3 = ""
        self.label10.config(text=self.arg3)




    #
    # Loads the rules in from the file, repopulating self.rulesDict
    #
    def loadRulesFromFile(self, filenameIn):
        # Clear old rules
        self.rulesList.clear()

        # load file
        with open(filenameIn,'r') as f:
            rulesTemp = [x.strip().split('\t') for x in f]

        # repopulate rules
        for rule in rulesTemp:
            self.rulesList.append(rule)

        # sort
        self.rulesList.sort()

    #
    # Save current rules to file
    #
    def saveRulesToFile(self, filenameOut):
        file = open(filenameOut, 'w')

        for rule in self.rulesList:
            ruleName = rule[0]
            ruleStr = ruleName
            for ruleElem in rule[1:]:
                ruleStr += "\t" + ruleElem
            file.write(ruleStr + "\n")

        file.close()


    #
    # Add new rule the user has placed in the Entry boxes.
    #
    def addRule(self):
        # Parse rule
        ruleName = self.entry1.get().upper()
        ruleArgs = []

        arg0 = self.entry2.get().upper()
        arg1 = self.entry3.get().upper()
        arg2 = self.entry4.get().upper()
        arg3 = self.entry5.get().upper()

        if (len(arg0) > 0): ruleArgs.append(arg0)
        if (len(arg1) > 0): ruleArgs.append(arg1)
        if (len(arg2) > 0): ruleArgs.append(arg2)
        if (len(arg3) > 0): ruleArgs.append(arg3)

        # If rule meets minimum criteria (has at least one argument), then store rule
        if (len(ruleArgs) > 0):
            rule = []
            rule.append(ruleName)
            rule = rule + ruleArgs
            self.rulesList.append(rule)

            # Reset rule entry widgets
            self.entry1.delete(0, END)
            self.entry2.delete(0, END)
            self.entry3.delete(0, END)
            self.entry4.delete(0, END)
            self.entry5.delete(0, END)

            self.entry1.insert(0, "NAME")

            # Save rules
            self.saveRulesToFile(self.rulesFilename)

            # Repopulare rules
            self.populateRules()


    #
    # Repopulates the 'rules' listbox by reading the rules in from the rules file
    #
    def populateRules(self):
        # Clear existing listbox content
        self.listbox.delete(0, END)

        # Load rules
        self.loadRulesFromFile(self.rulesFilename)

        # Repopulate listbox
        for rule in self.rulesList:
            ruleName = rule[0]
            ruleStr = ruleName + ": "
            for ruleElem in rule[1:]:
                ruleStr += "\t[" + ruleElem + "] "
            self.listbox.insert(END, ruleStr)


def main():
    GridDemo().mainloop()

if __name__ == "__main__":
    main()
