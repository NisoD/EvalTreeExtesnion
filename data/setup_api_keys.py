#!/usr/bin/env python3
"""
API Setup and Testing Script

This script helps you set up API keys and test the question generation system.
"""

import os
import sys
import json

def setup_api_keys():
    """Interactive setup for API keys."""
    
    print("🔑 API KEY SETUP")
    print("=" * 50)
    
    print("\nThis system requires two API keys:")
    print("1. OpenAI API Key (for GPT-4o-mini question generation)")
    print("2. Together AI API Key (for Llama-3.1-8B evaluation)")
    
    # Check existing keys
    existing_openai = os.getenv("OPENAI_API_KEY")
    existing_together = os.getenv("TOGETHER_API_KEY")
    
    if existing_openai:
        print(f"\n✅ OpenAI key found: ...{existing_openai[-8:]}")
    else:
        print("\n❌ No OpenAI key found")
    
    if existing_together:
        print(f"✅ Together AI key found: ...{existing_together[-8:]}")
    else:
        print("❌ No Together AI key found")
    
    print("\n" + "="*50)
    print("SETUP OPTIONS:")
    print("1. Set keys as environment variables (recommended)")
    print("2. Pass keys as command line arguments")
    print("3. Create a .env file")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        print("\n📝 Add these lines to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
        print("export OPENAI_API_KEY='your_openai_key_here'")
        print("export TOGETHER_API_KEY='your_together_key_here'")
        print("\nThen restart your terminal or run: source ~/.bashrc")
        
    elif choice == "2":
        print("\n📝 Use command line arguments:")
        print("python data/weakness_question_generator.py \\")
        print("  --profile data/llama_8b_weakness_profile.json \\")
        print("  --output weakness_validation \\")
        print("  --openai-key 'your_openai_key' \\")
        print("  --together-key 'your_together_key' \\")
        print("  --num-questions 2")
        
    elif choice == "3":
        env_content = """# API Keys for Weakness Question Generator
OPENAI_API_KEY=your_openai_key_here
TOGETHER_API_KEY=your_together_key_here
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ Created .env file")
        print("📝 Edit .env and add your actual API keys")
        print("💡 Install python-dotenv: pip install python-dotenv")
        print("💡 Load in your script with: from dotenv import load_dotenv; load_dotenv()")

def test_api_setup():
    """Test if APIs are working."""
    
    print("\n🧪 TESTING API SETUP")
    print("=" * 50)
    
    # Test OpenAI
    try:
        import openai
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            openai.api_key = openai_key
            print("✅ OpenAI key loaded")
            
            # Simple test
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello, respond with just 'API test successful'"}],
                    max_tokens=10
                )
                print("✅ OpenAI API test successful")
            except Exception as e:
                print(f"❌ OpenAI API test failed: {e}")
        else:
            print("❌ No OpenAI key found")
            
    except ImportError:
        print("❌ OpenAI library not installed. Run: pip install openai")
    
    # Test Together AI
    try:
        from together import Together
        together_key = os.getenv("TOGETHER_API_KEY")
        
        if together_key:
            client = Together(api_key=together_key)
            print("✅ Together AI key loaded")
            
            # Simple test
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
                    messages=[{"role": "user", "content": "Hello, respond with just 'API test successful'"}],
                    max_tokens=10
                )
                print("✅ Together AI API test successful")
            except Exception as e:
                print(f"❌ Together AI API test failed: {e}")
        else:
            print("❌ No Together AI key found")
            
    except ImportError:
        print("❌ Together library not installed. Run: pip install together")

def show_usage_examples():
    """Show usage examples."""
    
    print("\n📚 USAGE EXAMPLES")
    print("=" * 50)
    
    print("\n1. Generate questions only (if you only have OpenAI key):")
    print("""python data/weakness_question_generator.py \\
  --profile data/llama_8b_weakness_profile.json \\
  --output weakness_validation \\
  --num-questions 2 \\
  --generate-only""")
    
    print("\n2. Full pipeline (generation + evaluation):")
    print("""python data/weakness_question_generator.py \\
  --profile data/llama_8b_weakness_profile.json \\
  --output weakness_validation \\
  --num-questions 2""")
    
    print("\n3. Test with different weakness profiles:")
    print("""# Test with threshold 0.5 profile
python data/weakness_question_generator.py \\
  --profile data/llama_8b_weakness_profile_t05.json \\
  --output weakness_validation_t05 \\
  --num-questions 3""")
    
    print("\n📊 Expected outputs:")
    print("- weakness_validation_generated_questions.json")
    print("- weakness_validation_evaluation_results.json") 
    print("- weakness_validation_validation_analysis.json")

def check_dependencies():
    """Check if required libraries are installed."""
    
    print("\n📦 DEPENDENCY CHECK")
    print("=" * 50)
    
    required_libs = [
        ("openai", "OpenAI API client"),
        ("together", "Together AI client"),
        ("json", "JSON handling (built-in)"),
        ("time", "Time utilities (built-in)"),
        ("random", "Random sampling (built-in)")
    ]
    
    missing = []
    
    for lib, description in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib}: {description}")
        except ImportError:
            print(f"❌ {lib}: {description} - MISSING")
            missing.append(lib)
    
    if missing:
        print(f"\n📥 Install missing libraries:")
        installable = [lib for lib in missing if lib not in ['json', 'time', 'random']]
        if installable:
            print(f"pip install {' '.join(installable)}")
    else:
        print("\n✅ All dependencies satisfied!")

def main():
    """Main setup function."""
    
    print("🚀 WEAKNESS QUESTION GENERATOR SETUP")
    print("=" * 60)
    
    # Check dependencies first
    check_dependencies()
    
    # Setup API keys
    setup_api_keys()
    
    # Test APIs if keys are available
    if os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY"):
        test_api_setup()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Get your API keys from:")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Together AI: https://api.together.xyz/settings/api-keys")
    print("2. Set up the keys using one of the methods above")
    print("3. Run the question generator on your weakness profile")
    print("4. Analyze the results to validate weakness predictions!")
    
    print(f"\n✨ Ready to validate your weakness discoveries! ✨")

if __name__ == "__main__":
    main() 