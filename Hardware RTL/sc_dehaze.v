module sc_dehaze (
    input clk,
    input [7:0] pixel,
    output reg [127:0] bitstream
);

    reg [127:0] sobol = 128'hA5F0_1234_5678_9ABC_DEF0_AAAA_5555_0F0F;

    integer i;

    reg [127:0] smooth;
    reg [127:0] sharpen;

    always @(posedge clk) begin

        //change sobol every cycle
        sobol <= {sobol[126:0], sobol[127] ^ sobol[125]};

        //stochastic encoding
        for (i = 0; i < 128; i = i + 1) begin
            bitstream[i] <= (pixel > sobol[i]);
        end

        //smoothing
        smooth <= (bitstream >> 1) ^ (bitstream << 1);

        //sharpening
        sharpen <= bitstream ^ (~smooth);

        bitstream <= sharpen;
    end

endmodule