#!/usr/bin/env python3
"""
Simple Debug Sunburst - Test if Plotly works at all
"""

import plotly.graph_objects as go
import plotly.offline as pyo

def create_minimal_test():
    """Create the most minimal sunburst possible"""
    print("Creating minimal test sunburst...")
    
    # Super simple data
    fig = go.Figure(go.Sunburst(
        ids=["A", "B", "C"],
        labels=["Physics", "Math", "Chemistry"],
        parents=["", "", ""],
        values=[10, 20, 15]
    ))
    
    fig.update_layout(
        title="Minimal Test Sunburst",
        width=400,
        height=400
    )
    
    # Save with explicit offline mode
    pyo.plot(fig, filename="minimal_test.html", auto_open=False)
    print("✅ Minimal test saved to minimal_test.html")

def create_simple_bar_chart():
    """Create a simple bar chart to test if Plotly works at all"""
    print("Creating simple bar chart...")
    
    import plotly.express as px
    
    # Simple bar chart
    fig = px.bar(
        x=["Physics", "Math", "Chemistry"],
        y=[0.2, 0.3, 0.4],
        title="Simple Test Chart",
        labels={"x": "Subject", "y": "Score"}
    )
    
    fig.update_layout(width=500, height=400)
    
    pyo.plot(fig, filename="simple_bar_test.html", auto_open=False)
    print("✅ Simple bar chart saved to simple_bar_test.html")

def create_basic_html():
    """Create basic HTML without Plotly to test browser"""
    print("Creating basic HTML test...")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Basic HTML Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .box { 
            width: 200px; 
            height: 100px; 
            background-color: #4CAF50; 
            color: white; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>🔍 Browser Test</h1>
    <p>If you can see this, your browser works!</p>
    <div class="box">Green Box Test</div>
    
    <h2>JavaScript Test</h2>
    <p id="js-test">JavaScript is NOT working</p>
    
    <script>
        document.getElementById('js-test').innerHTML = '✅ JavaScript IS working!';
        document.getElementById('js-test').style.color = 'green';
        console.log('JavaScript is working in console');
    </script>
</body>
</html>
"""
    
    with open("basic_html_test.html", "w") as f:
        f.write(html_content)
    
    print("✅ Basic HTML test saved to basic_html_test.html")

def create_manual_sunburst():
    """Create sunburst with manual HTML/JavaScript"""
    print("Creating manual sunburst...")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Manual Sunburst Test</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #sunburst { width: 100%; height: 500px; }
    </style>
</head>
<body>
    <h1>🌞 Manual Sunburst Test</h1>
    <div id="sunburst"></div>
    
    <script>
        console.log('Starting manual sunburst creation...');
        
        var data = [{
            type: "sunburst",
            ids: ["A", "B", "C", "A1", "A2", "B1"],
            labels: ["Physics<br>0.2", "Math<br>0.3", "Chemistry<br>0.4", "Quantum", "Classical", "Organic"],
            parents: ["", "", "", "A", "A", "B"],
            values: [10, 20, 15, 5, 5, 10],
            branchvalues: "total",
            marker: {
                colorscale: 'RdYlBu_r',
                colors: [1, 0.5, 0, 0.8, 0.2, 0.6]
            }
        }];
        
        var layout = {
            title: "Manual Test Sunburst",
            width: 500,
            height: 500
        };
        
        Plotly.newPlot('sunburst', data, layout)
            .then(function() {
                console.log('✅ Sunburst created successfully!');
            })
            .catch(function(error) {
                console.error('❌ Sunburst creation failed:', error);
                document.getElementById('sunburst').innerHTML = '<p style="color:red;">Failed to create sunburst: ' + error + '</p>';
            });
    </script>
</body>
</html>
"""
    
    with open("manual_sunburst_test.html", "w") as f:
        f.write(html_content)
    
    print("✅ Manual sunburst test saved to manual_sunburst_test.html")

def main():
    print("🔍 Creating Debug Tests...")
    print("="*50)
    
    try:
        # Test 1: Basic HTML
        create_basic_html()
        
        # Test 2: Simple bar chart
        create_simple_bar_chart()
        
        # Test 3: Minimal sunburst
        create_minimal_test()
        
        # Test 4: Manual sunburst
        create_manual_sunburst()
        
        print("\n" + "="*50)
        print("🔍 DEBUG TESTS COMPLETE!")
        print("="*50)
        print("Created test files:")
        print("  1. basic_html_test.html - Test if browser/JavaScript works")
        print("  2. simple_bar_test.html - Test if Plotly works (bar chart)")
        print("  3. minimal_test.html - Test minimal sunburst")
        print("  4. manual_sunburst_test.html - Manual sunburst with CDN")
        
        print("\n📋 Testing Steps:")
        print("1. Open basic_html_test.html first")
        print("   - Should show green box and 'JavaScript IS working'")
        print("2. Open simple_bar_test.html")
        print("   - Should show a simple bar chart")
        print("3. Open minimal_test.html")
        print("   - Should show minimal sunburst")
        print("4. Open manual_sunburst_test.html")
        print("   - Uses CDN version of Plotly")
        
        print("\n🔧 Debugging:")
        print("- Press F12 in browser → Console tab")
        print("- Look for error messages")
        print("- Try different browsers")
        
    except Exception as e:
        print(f"❌ Error creating debug tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 