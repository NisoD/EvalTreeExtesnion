#!/usr/bin/env python3
"""
Debug Sunburst Issue - Find out why HTML is empty
"""

import json
import plotly.graph_objects as go
import hashlib

def load_shortened_tree():
    """Load the shortened tree JSON"""
    print("Loading shortened tree...")
    
    try:
        with open('shortened tree.json', 'r') as f:
            content = f.read()
            tree = json.loads(content)
    except json.JSONDecodeError:
        with open('shortened tree.json', 'r') as f:
            content = f.read()
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
            tree = json.loads(content)
    
    return tree

def create_simple_test_sunburst():
    """Create the simplest possible sunburst to test if Plotly works"""
    print("\n🔍 TEST 1: Creating simple test sunburst...")
    
    fig = go.Figure(go.Sunburst(
        ids=["A", "B", "C"],
        labels=["Category A", "Category B", "Category C"],
        parents=["", "", ""],
        values=[10, 20, 30],
    ))
    
    fig.update_layout(title="Simple Test Sunburst")
    fig.write_html("test_simple_sunburst.html")
    print("✅ Simple test saved to test_simple_sunburst.html")

def create_manual_data_sunburst():
    """Create sunburst with manually crafted data"""
    print("\n🔍 TEST 2: Creating sunburst with manual data...")
    
    # Manual data that should definitely work
    data = [
        {
            'ids': 'physics',
            'labels': 'Physics<br>0.200',
            'parents': '',
            'values': 10
        },
        {
            'ids': 'math', 
            'labels': 'Math<br>0.300',
            'parents': '',
            'values': 20
        },
        {
            'ids': 'chemistry',
            'labels': 'Chemistry<br>0.400', 
            'parents': '',
            'values': 15
        }
    ]
    
    print("Manual data:")
    for item in data:
        print(f"  ID: {item['ids']}, Label: {item['labels']}, Parent: '{item['parents']}', Value: {item['values']}")
    
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in data],
        labels=[d['labels'] for d in data],
        parents=[d['parents'] for d in data],
        values=[d['values'] for d in data],
        marker=dict(
            colors=[0.2, 0.5, 0.8],
            colorscale='RdYlBu_r'
        )
    ))
    
    fig.update_layout(title="Manual Data Sunburst")
    fig.write_html("test_manual_sunburst.html")
    print("✅ Manual data sunburst saved to test_manual_sunburst.html")

def extract_and_debug_real_data():
    """Extract real data and debug the structure"""
    print("\n🔍 TEST 3: Extracting and debugging real data...")
    
    tree = load_shortened_tree()
    
    # Extract just first few items for debugging
    debug_data = []
    
    def extract_debug_items(node, level=0, parent_id=""):
        if not isinstance(node, dict) or level > 2 or len(debug_data) >= 5:
            return
        
        capability = node.get('capability', f'Node_{level}')
        weakness_stats = node.get('weakness_stats', {})
        
        if weakness_stats and weakness_stats.get('question_count', 0) >= 3:
            mean_score = weakness_stats.get('mean_dove_score', 0.5)
            question_count = weakness_stats.get('question_count', 0)
            
            # Simple ID
            simple_id = f"node_{len(debug_data)}"
            
            item = {
                'ids': simple_id,
                'labels': f"{capability[:20]}<br>{mean_score:.3f}",
                'parents': parent_id,
                'values': question_count
            }
            
            debug_data.append(item)
            print(f"  Added: ID='{item['ids']}', Label='{item['labels']}', Parent='{item['parents']}', Value={item['values']}")
            
            # Add one child if possible
            if 'subtrees' in node and isinstance(node['subtrees'], list) and len(node['subtrees']) > 0:
                child_node = node['subtrees'][0]
                extract_debug_items(child_node, level + 1, simple_id)
        
        # Try siblings
        if 'subtrees' in node and isinstance(node['subtrees'], list):
            for subtree in node['subtrees'][1:]:
                if len(debug_data) < 5:
                    extract_debug_items(subtree, level, parent_id)
    
    extract_debug_items(tree)
    
    print(f"\nExtracted {len(debug_data)} debug items:")
    for i, item in enumerate(debug_data):
        print(f"  {i+1}. {item}")
    
    if len(debug_data) == 0:
        print("❌ No debug data extracted!")
        return
    
    # Check for issues
    ids = [d['ids'] for d in debug_data]
    parents = [d['parents'] for d in debug_data]
    
    print(f"\n🔍 Data validation:")
    print(f"  IDs: {ids}")
    print(f"  Parents: {parents}")
    print(f"  Unique IDs: {len(set(ids)) == len(ids)}")
    print(f"  Empty parents count: {parents.count('')}")
    
    # Create sunburst with debug data
    print(f"\n🔍 Creating sunburst with {len(debug_data)} debug items...")
    
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=[d['labels'] for d in debug_data],
        parents=parents,
        values=[d['values'] for d in debug_data],
        marker=dict(
            colors=[0.2, 0.4, 0.6, 0.8, 1.0][:len(debug_data)],
            colorscale='RdYlBu_r'
        )
    ))
    
    fig.update_layout(title=f"Debug Real Data Sunburst ({len(debug_data)} items)")
    fig.write_html("test_debug_real_sunburst.html")
    print("✅ Debug real data sunburst saved to test_debug_real_sunburst.html")

def check_html_files():
    """Check if HTML files actually contain sunburst data"""
    print("\n🔍 TEST 4: Checking HTML file contents...")
    
    import os
    
    html_files = [
        "test_simple_sunburst.html",
        "test_manual_sunburst.html", 
        "test_debug_real_sunburst.html",
        "working_shortened_sunburst.html"
    ]
    
    for filename in html_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  {filename}: {size:,} bytes")
            
            # Check if it contains sunburst data
            with open(filename, 'r') as f:
                content = f.read()
                has_sunburst = 'Sunburst' in content
                has_plotly = 'Plotly' in content
                has_data = '"ids"' in content or '"labels"' in content
                
                print(f"    Contains Sunburst: {has_sunburst}")
                print(f"    Contains Plotly: {has_plotly}")
                print(f"    Contains Data: {has_data}")
                
                if not has_data:
                    print(f"    ❌ {filename} appears to be empty of data!")
                else:
                    print(f"    ✅ {filename} appears to contain data")
        else:
            print(f"  {filename}: NOT FOUND")

def main():
    print("🚨 DEBUGGING EMPTY SUNBURST ISSUE")
    print("="*60)
    
    # Test 1: Simple sunburst
    create_simple_test_sunburst()
    
    # Test 2: Manual data sunburst
    create_manual_data_sunburst()
    
    # Test 3: Real data extraction and debugging
    extract_and_debug_real_data()
    
    # Test 4: Check HTML files
    check_html_files()
    
    print("\n" + "="*60)
    print("🔍 DEBUG COMPLETE!")
    print("="*60)
    print("Check these test files:")
    print("  • test_simple_sunburst.html")
    print("  • test_manual_sunburst.html") 
    print("  • test_debug_real_sunburst.html")
    print("\nIf ALL of these are empty, the issue is with Plotly/browser.")
    print("If SOME work, the issue is with our data structure.")

if __name__ == "__main__":
    main() 