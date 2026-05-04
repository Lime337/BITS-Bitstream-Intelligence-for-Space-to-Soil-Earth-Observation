module popcount #(parameter D = 4096) (
    input clk,
    input [D-1:0] in,
    output reg [11:0] count
);

    //stage registers
    reg [2047:0] s1;
    reg [1023:0] s2;
    reg [511:0]  s3;
    reg [255:0]  s4;
    reg [127:0]  s5;
    reg [63:0]   s6;
    reg [31:0]   s7;
    reg [15:0]   s8;
    reg [7:0]    s9;
    reg [3:0]    s10;
    reg [1:0]    s11;
    reg [11:0]   s12;

    integer i;

    //stage 1
    always @(posedge clk) begin
        for (i = 0; i < 2048; i = i + 1)
            s1[i] <= in[2*i] + in[2*i+1];
    end

    //stage 2
    always @(posedge clk) begin
        for (i = 0; i < 1024; i = i + 1)
            s2[i] <= s1[2*i] + s1[2*i+1];
    end

    //stage 3
    always @(posedge clk) begin
        for (i = 0; i < 512; i = i + 1)
            s3[i] <= s2[2*i] + s2[2*i+1];
    end

    //stage 4
    always @(posedge clk) begin
        for (i = 0; i < 256; i = i + 1)
            s4[i] <= s3[2*i] + s3[2*i+1];
    end

    //stage 5
    always @(posedge clk) begin
        for (i = 0; i < 128; i = i + 1)
            s5[i] <= s4[2*i] + s4[2*i+1];
    end

    //stage 6
    always @(posedge clk) begin
        for (i = 0; i < 64; i = i + 1)
            s6[i] <= s5[2*i] + s5[2*i+1];
    end

    //stage 7
    always @(posedge clk) begin
        for (i = 0; i < 32; i = i + 1)
            s7[i] <= s6[2*i] + s6[2*i+1];
    end

    //stage 8
    always @(posedge clk) begin
        for (i = 0; i < 16; i = i + 1)
            s8[i] <= s7[2*i] + s7[2*i+1];
    end

    //stage 9
    always @(posedge clk) begin
        for (i = 0; i < 8; i = i + 1)
            s9[i] <= s8[2*i] + s8[2*i+1];
    end

    //stage 10
    always @(posedge clk) begin
        for (i = 0; i < 4; i = i + 1)
            s10[i] <= s9[2*i] + s9[2*i+1];
    end

    //stage 11
    always @(posedge clk) begin
        for (i = 0; i < 2; i = i + 1)
            s11[i] <= s10[2*i] + s10[2*i+1];
    end

    //final stage
    always @(posedge clk) begin
        s12 <= s11[0] + s11[1];
        count <= s12;
    end

endmodule