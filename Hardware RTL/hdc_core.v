(* dont_touch = "true" *)
module hdc_core (
    input clk,
    input [127:0] bitstream,
    output reg class_out
);

    parameter D = 4096;
    parameter BASE = 128;

    reg [BASE-1:0] base_hv = 128'hA5A5_F0F0_1234_5678_9ABC_DEF0_0F0F_F0F0;

    (* keep = "true" *) reg [D-1:0] hv;
    (* keep = "true" *) reg [D-1:0] class0;
    (* keep = "true" *) reg [D-1:0] class1;

    integer i;

    always @(posedge clk) begin
        for (i = 0; i < D; i = i + 1) begin
            hv[i] <= bitstream[i % BASE];
            class0[i] <= base_hv[i % BASE];
            class1[i] <= ~base_hv[i % BASE];
        end
    end

    wire [11:0] sim0;
    wire [11:0] sim1;

    hamming_sim #(D) h0 (.clk(clk), .a(hv), .b(class0), .sim(sim0));
    hamming_sim #(D) h1 (.clk(clk), .a(hv), .b(class1), .sim(sim1));

    //delay
    reg [11:0] sim0_d, sim1_d;

    always @(posedge clk) begin
        sim0_d <= sim0;
        sim1_d <= sim1;
    end

    always @(posedge clk) begin
        class_out <= (sim1_d > sim0_d);
    end

endmodule