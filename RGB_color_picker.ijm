macro "Color Picker Tool -C44f-o4499" {
        getCursorLoc(x, y, z, flags);
        v = getPixel(x,y);
        row = nResults;
        setResult("X", row, x);
        setResult("Y", row, y);
        if (nSlices>1) setResult("Z", row, z);
        if (bitDepth==24) {
            red = (v>>16)&0xff;  // extract red byte (bits 23-17)
            green = (v>>8)&0xff; // extract green byte (bits 15-8)
            blue = v&0xff;       // extract blue byte (bits 7-0)
            setResult("Red", row, red);
            setResult("Green", row, green);
            setResult("Blue", row, blue);
        } else
            setResult("Value", row, v);
        updateResults;
   }
