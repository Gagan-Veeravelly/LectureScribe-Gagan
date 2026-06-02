from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript():
    video_id = "7sB052Pz0sQ" # MIT Deep Learning
    print(f"📥 Connecting to YouTube API for ID: {video_id}...")
    
    try:
        # Fetch the transcript object
        transcript = YouTubeTranscriptApi().fetch(video_id)
        
        # --- DEBUG: Check what we actually got ---
        if transcript:
            first_item = transcript[0]
            # print(f"Debug: Item type is {type(first_item)}")
            # print(f"Debug: Item attributes: {dir(first_item)}")
        
        # --- THE FIX ---
        # We access '.text' as an attribute, not a dictionary key
        full_text = " ".join([t.text for t in transcript])
        
        print(f"✅ SUCCESS! Downloaded {len(full_text)} characters.")
        print("--- Preview ---")
        print(full_text[:300] + "...")
        
        # Save to file
        with open("transcript.txt", "w") as f:
            f.write(full_text)
        print("💾 Saved to 'transcript.txt'")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    get_transcript()
