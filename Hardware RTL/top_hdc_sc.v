module top_hdc_sc (
    input clk,
    input rst,
    output reg [7:0] debug_out
);

    reg [7:0] pixel_reg;

    always @(posedge clk) begin
        if (rst)
            pixel_reg <= 0;
        else
            pixel_reg <= pixel_reg + 1;
    end

    wire [127:0] sc_stream;
    wire class_out;

    sc_dehaze sc_block (
        .clk(clk),
        .pixel(pixel_reg),
        .bitstream(sc_stream)
    );

    hdc_core hdc_block (
        .clk(clk),
        .bitstream(sc_stream),
        .class_out(class_out)
    );

    always @(posedge clk) begin
        debug_out <= {7'b0, class_out};
    end

endmodule