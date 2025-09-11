# Typing speed test simple
import time, random
SAMPLES=['the quick brown fox','hello world','lorem ipsum dolor sit amet']
s=random.choice(SAMPLES); print('Type this:'); print(s); input('Ready? press enter'); t0=time.time(); inp=input(); t1=time.time()
correct = inp.strip()==s
wpm = (len(inp.split())/(t1-t0))*60 if (t1-t0)>0 else 0
print('Correct:',correct,'WPM:',int(wpm))
